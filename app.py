import streamlit as st
# load .env for local development (optional); re-run to ensure latest values are taken after restart
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except Exception:
    pass
import sounddevice as sd
import soundfile as sf
import threading
import time
import random
import os
import io
import smtplib
from email.message import EmailMessage
from datetime import datetime

from pipeline import add_to_pipeline, get_result


def _get_env(name, default=None):
    try:
        val = os.environ.get(name, default)
        if isinstance(val, str):
            return val.strip()
        return val
    except Exception:
        return default

def _resolve_config(keys, default=""):
    """Return the first non-empty value for any of the provided keys.

    Checks os.environ first, then Streamlit secrets if available. Trims strings.
    """
    try:
        lower_map = {k.lower(): k for k in os.environ.keys()}
        for key in keys:
            try:
                # direct lookup first
                val = os.environ.get(key)
                if not (isinstance(val, str) and val.strip()) and val in (None, ""):
                    # case-insensitive lookup
                    ci = lower_map.get(str(key).lower())
                    if ci:
                        val = os.environ.get(ci)
                if isinstance(val, str) and val.strip():
                    return val.strip()
                if val not in (None, "") and not isinstance(val, str):
                    return val
            except Exception:
                pass
        try:
            if st.secrets:
                # Build case-insensitive view of secrets too
                try:
                    secrets_items = list(st.secrets.items())
                except Exception:
                    secrets_items = []
                secrets_ci = {str(k).lower(): v for k, v in secrets_items}
                for key in keys:
                    try:
                        val = st.secrets.get(key)
                        if val in (None, ""):
                            val = secrets_ci.get(str(key).lower())
                        if isinstance(val, str) and val.strip():
                            return str(val).strip()
                        if val not in (None, "") and not isinstance(val, str):
                            return val
                    except Exception:
                        pass
        except Exception:
            pass
    except Exception:
        pass
    return default

# If the user supplied a Gemini key via Streamlit secrets, set it into env so
# backend modules (summarization_module) can read it. This keeps secrets out of
# source control while allowing the summarizer to detect the key.
try:
    if st.secrets and st.secrets.get("GOOGLE_API_KEY"):
        import os
        os.environ.setdefault("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY"))
    if st.secrets and st.secrets.get("GOOGLE_GEMINI_API_KEY"):
        import os
        os.environ.setdefault("GOOGLE_GEMINI_API_KEY", st.secrets.get("GOOGLE_GEMINI_API_KEY"))
    # Also allow setting SMTP/email variables via Streamlit secrets
    if st.secrets:
        for key in (
            "SMTP_SERVER",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "SMTP_USE_SSL",
            "EMAIL_RECIPIENT",
            "EMAIL_AUTO_SEND",
        ):
            try:
                val = st.secrets.get(key)
                if val is not None:
                    os.environ.setdefault(key, str(val))
            except Exception:
                pass
except Exception:
    # st.secrets may not be available in some runtimes; ignore failures
    pass


st.set_page_config(page_title="AI Live Meeting Summarizer", layout="wide")
st.title("AI Live Meeting Summarizer")


def safe_rerun():
    """Try to trigger a Streamlit rerun in a backwards-compatible way.

    Some Streamlit versions don't expose `st.experimental_rerun`. We first
    try to call it; if it's missing, toggle a query param using
    `experimental_set_query_params` which forces the app to rerun in the
    browser.
    """
    # preferred API (available in many Streamlit versions)
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        # continue to fallback
        pass

    # fallback: toggle a query param to force a rerun in the browser
    try:
        qp = {}
        try:
            qp = st.experimental_get_query_params() or {}
        except Exception:
            qp = {}
        v = int(qp.get("_r", ["0"])[0]) if qp.get("_r") else 0
        qp["_r"] = [str(v + 1)]
        try:
            st.experimental_set_query_params(**qp)
        except Exception:
            # last resort: do nothing
            return
    except Exception:
        # If anything goes wrong in the fallback, give up silently
        return

# --- Configuration ---
FS = 16000
CHANNELS = 1
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

if "recording" not in st.session_state:
    st.session_state.recording = False
if "current_file" not in st.session_state:
    st.session_state.current_file = None
if "status" not in st.session_state:
    st.session_state.status = "idle"  # idle, recording, processing, done
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "last_progress_time" not in st.session_state:
    st.session_state.last_progress_time = 0.0
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Recover last in-progress job across reloads
try:
    last_job_path = os.path.join("results", "last_job.json")
    if st.session_state.current_file is None and os.path.exists(last_job_path):
        import json as _json
        with open(last_job_path, 'r', encoding='utf-8') as _lf:
            obj = _json.load(_lf)
        audio_file_path = obj.get('audio_file') if isinstance(obj, dict) else None
        if audio_file_path and os.path.exists(audio_file_path):
            st.session_state.current_file = audio_file_path
            st.session_state.status = obj.get('status', 'processing')
except Exception:
    pass


def _record_thread(file_path):
    """Background thread that records audio to `file_path` until stop flag."""
    st.session_state.status = "recording"
    try:
        with sf.SoundFile(file_path, mode="w", samplerate=FS, channels=CHANNELS, subtype="PCM_16") as f:
            def callback(indata, frames, time_info, status):
                if status:
                    print("Recording status:", status)
                # indata is a numpy array with shape (frames, channels)
                f.write(indata.copy())
                if not st.session_state.recording:
                    raise sd.CallbackStop()

            with sd.InputStream(samplerate=FS, channels=CHANNELS, callback=callback, dtype="int16"):
                while st.session_state.recording:
                    time.sleep(0.1)
    except Exception as e:
        # If the callback requested stop via CallbackStop this is normal
        print("Recorder stopped:", e)
    finally:
        st.session_state.recording = False
        st.session_state.status = "processing"
        st.session_state.progress = 0
        st.session_state.last_progress_time = time.time()
        # enqueue for background processing
        add_to_pipeline(file_path)
        # persist job so refresh doesn't lose context
        try:
            os.makedirs("results", exist_ok=True)
            import json as _json
            with open(os.path.join("results", "last_job.json"), 'w', encoding='utf-8') as _lf:
                _json.dump({"audio_file": file_path, "status": "processing"}, _lf)
        except Exception:
            pass


def start_recording():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDINGS_DIR, f"meeting_{timestamp}.wav")
    st.session_state.current_file = filename
    st.session_state.recording = True
    # persist job so refresh doesn't lose context
    try:
        os.makedirs("results", exist_ok=True)
        import json as _json
        with open(os.path.join("results", "last_job.json"), 'w', encoding='utf-8') as _lf:
            _json.dump({"audio_file": filename, "status": "recording"}, _lf)
    except Exception:
        pass
    thread = threading.Thread(target=_record_thread, args=(filename,), daemon=True)
    thread.start()


def stop_recording():
    st.session_state.recording = False


def send_email(recipient, subject, body, attachment_bytes=None, attachment_name=None, smtp_server=None, smtp_port=587, username=None, password=None, use_ssl=None):
    # Ensure body is string
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8", errors="ignore")
    msg = EmailMessage()
    msg["From"] = username or os.environ.get("SMTP_USERNAME") or "noreply@example.com"
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)
    if attachment_bytes and attachment_name:
        msg.add_attachment(attachment_bytes, maintype="application", subtype="octet-stream", filename=attachment_name)
    env_ssl = str(os.environ.get("SMTP_USE_SSL", "false")).lower() in ("1","true","yes","y")
    use_ssl = env_ssl if use_ssl is None else bool(use_ssl)
    if use_ssl:
        with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as s:
            if username and password:
                s.login(username, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as s:
            try:
                s.ehlo()
                s.starttls()
                s.ehlo()
            except Exception:
                pass
            if username and password:
                s.login(username, password)
            s.send_message(msg)

def _test_smtp_connection(smtp_server, smtp_port, username=None, password=None, use_ssl=False):
    """Return (ok: bool, message: str) after attempting an SMTP connect/login."""
    try:
        if use_ssl:
            with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as s:
                s.ehlo()
                if username and password:
                    s.login(username, password)
        else:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as s:
                s.ehlo()
                s.starttls()
                s.ehlo()
                if username and password:
                    s.login(username, password)
        return True, "SMTP connection/login succeeded"
    except Exception as e:
        return False, f"SMTP test failed: {e}"


# --- Layout ---
left, right = st.columns([2, 3])

with left:
    st.header("Controls")
    st.subheader("Upload audio file")
    uploaded = st.file_uploader("Upload a WAV/MP3/FLAC file", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=False)
    if uploaded is not None:
        # Save uploaded file to recordings dir and enqueue
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = os.path.basename(uploaded.name)
        save_path = os.path.join(RECORDINGS_DIR, f"upload_{timestamp}_{safe_name}")
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.session_state.current_file = save_path
        st.session_state.status = "processing"
        st.session_state.progress = 0
        st.session_state.last_progress_time = time.time()
        add_to_pipeline(save_path)
        # persist job so refresh doesn't lose context
        try:
            os.makedirs("results", exist_ok=True)
            import json as _json
            with open(os.path.join("results", "last_job.json"), 'w', encoding='utf-8') as _lf:
                _json.dump({"audio_file": save_path, "status": "processing"}, _lf)
        except Exception:
            pass
        st.success(f"Uploaded and enqueued: {save_path}")

    if not st.session_state.recording:
        if st.button("Start Recording", key="start"):
            start_recording()
    else:
        if st.button("Stop Recording", key="stop"):
            stop_recording()

    st.markdown("---")
    st.write("Status:", st.session_state.status)
    if st.session_state.recording:
        st.info("Recording... speak now")

    st.markdown("---")
    if st.session_state.current_file:
        st.write("Last file:", st.session_state.current_file)
        # show pipeline status if available
        status_path = st.session_state.current_file + ".status"
        try:
            if os.path.exists(status_path):
                st.write("Processing status:", open(status_path, "r").read())
        except Exception:
            pass
        # live transcription log (from results/transcripts/<base>_stt.txt)
        try:
            base = os.path.basename(st.session_state.current_file).replace('.wav', '')
            live_txt = os.path.join('results', 'transcripts', f"{base}_stt.txt")
            if os.path.isfile(live_txt) and st.session_state.status in ("processing", "done"):
                st.subheader("Live Transcription Log")
                st.text_area("live_transcript", value=open(live_txt, 'r', encoding='utf-8').read(), height=150)
        except Exception:
            pass

    if st.button("Refresh Result"):
        # simple manual refresh; Streamlit will rerun and check for results
        safe_rerun()

    # Email UI removed by request

with right:
    st.header("Live / Results")

    result = get_result()
    if result:
        st.session_state.status = "done"
        transcript, diarized_transcript, summary = result
        # persist latest result to survive reruns (e.g., after downloads)
        st.session_state.last_result = (transcript, diarized_transcript, summary)

        tabs = st.tabs(["Transcript", "Diarized", "Summary", "Export", "Email"])

        with tabs[0]:
            st.text_area("transcript", value=transcript, height=260)

        with tabs[1]:
            diarized_text = "\n".join(diarized_transcript) if isinstance(diarized_transcript, (list, tuple)) else str(diarized_transcript)
            st.text_area("diarized", value=diarized_text, height=260)

        with tabs[2]:
            st.text_area("summary", value=summary, height=260)

        with tabs[3]:
            md = f"# Meeting Summary\n\n## Summary\n{summary}\n\n## Transcript\n{transcript}\n"
            st.download_button("Download Summary (MD)", md, file_name="meeting_summary.md")
            st.download_button("Download Transcript (TXT)", transcript, file_name="meeting_transcript.txt")
            st.download_button("Download Diarized (TXT)", diarized_text, file_name="meeting_diarized.txt")
            # Attempt RTTM locate
            try:
                base = os.path.basename(st.session_state.current_file).replace('.wav','')
                rttm_path = os.path.join('results','diarization', f"{base}.rttm")
                if os.path.isfile(rttm_path):
                    st.download_button("Download Diarization (RTTM)", open(rttm_path,'rb').read(), file_name=f"{base}.rttm")
            except Exception:
                pass

            # PDF export
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas

                def make_pdf_bytes(title, summary_text, transcript_text):
                    buf = io.BytesIO()
                    c = canvas.Canvas(buf, pagesize=letter)
                    width, height = letter
                    y = height - 40
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(40, y, title)
                    y -= 30
                    c.setFont("Helvetica", 10)
                    for line in ("Summary:",) + tuple(summary_text.splitlines()):
                        if y < 60:
                            c.showPage()
                            y = height - 40
                        c.drawString(40, y, line)
                        y -= 14
                    y -= 10
                    for line in ("Transcript:",) + tuple(transcript_text.splitlines()):
                        if y < 60:
                            c.showPage()
                            y = height - 40
                        c.drawString(40, y, line)
                        y -= 12
                    c.save()
                    buf.seek(0)
                    return buf.read()

                pdf_bytes = make_pdf_bytes("Meeting Summary", summary, transcript)
                st.download_button("Download PDF", pdf_bytes, file_name="meeting_summary.pdf", mime="application/pdf")
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")

        with tabs[4]:
            st.write("Send the summary via email (uses environment variables)")
            # Prevent accidental UI emptying on rerun by keeping last_result
            keep = st.session_state.get('last_result')
            if keep and st.session_state.get('status') == 'done':
                st.session_state.last_result = keep
            if st.button("Send email"):
                try:
                    recipient = _resolve_config(["EMAIL_RECIPIENT", "RECIPIENT_EMAIL", "MAIL_TO"], "").strip()
                    # Support comma/semicolon separated list; take first for validation, send to all
                    recipients = [r.strip() for r in recipient.replace(';', ',').split(',') if r.strip()]
                    smtp = _resolve_config(["SMTP_SERVER", "MAIL_SERVER"], "").strip()
                    smtp_port_str = _resolve_config(["SMTP_PORT", "MAIL_PORT"], "587") or "587"
                    smtp_port = int(str(smtp_port_str).strip() or "587")
                    username = _resolve_config(["SMTP_USERNAME", "MAIL_USERNAME"], "").strip()
                    password = _resolve_config(["SMTP_PASSWORD", "MAIL_PASSWORD"], "")
                    use_ssl_val = _resolve_config(["SMTP_USE_SSL", "MAIL_USE_SSL"], "false")
                    use_ssl = str(use_ssl_val).lower() in ("1","true","yes","y", "on")

                    if not recipients:
                        st.error("EMAIL_RECIPIENT/RECIPIENT_EMAIL/MAIL_TO is not set")
                    elif not smtp:
                        st.error("SMTP_SERVER/MAIL_SERVER is not set")
                    else:
                        subject = f"Meeting Summary – {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        md = f"# Meeting Summary\n\n## Summary\n{summary}\n\n## Transcript\n{transcript}\n"
                        # Send to comma-separated list one by one for clearer failures
                        for r in recipients:
                            send_email(r, subject, md, attachment_bytes=md.encode("utf-8"), attachment_name="summary.md", smtp_server=smtp, smtp_port=smtp_port, username=username, password=password, use_ssl=use_ssl)
                        st.success("Email sent successfully")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
    else:
        # If no new result pulled from queue, show the last one if available (prevents UI emptying on rerun)
        if st.session_state.last_result and st.session_state.status == "done":
            transcript, diarized_transcript, summary = st.session_state.last_result
            st.subheader("Transcript")
            st.text_area("transcript", value=transcript, height=200)
            st.subheader("Diarized Transcript")
            st.text_area("diarized", value="\n".join(diarized_transcript) if isinstance(diarized_transcript, (list, tuple)) else str(diarized_transcript), height=200)
            st.subheader("Summary")
            st.text_area("summary", value=summary, height=200)
            md = f"# Meeting Summary\n\n## Summary\n{summary}\n\n## Transcript\n{transcript}\n"
            st.download_button("Download Summary (MD)", md, file_name="meeting_summary.md")
            st.download_button("Download Transcript (TXT)", transcript, file_name="meeting_transcript.txt")
        
        if st.session_state.status == "processing":
            # Light auto-refresh: increment a counter to trigger rerun without a hard browser refresh
            try:
                st.session_state["_auto_tick"] = st.session_state.get("_auto_tick", 0) + 1
                # Small empty element forces Streamlit to detect a change and rerun periodically
                st.caption(f"Updating… {st.session_state['_auto_tick']}")
            except Exception:
                pass

            # Show an estimated progress bar and spinner while the background worker runs
            # try reading a structured status file next to the current file
            status_path = None
            if st.session_state.current_file:
                status_path = st.session_state.current_file + ".status"

            percent = None
            phase = None
            if status_path and os.path.exists(status_path):
                try:
                    import json as _json
                    with open(status_path, 'r', encoding='utf-8') as sf:
                        obj = _json.load(sf)
                    percent = int(obj.get('percent')) if obj and obj.get('percent') is not None else None
                    phase = obj.get('phase') if obj and obj.get('phase') else None
                except Exception:
                    # not JSON / older file format: read textual tag
                    try:
                        text = open(status_path, 'r', encoding='utf-8').read().strip()
                        phase = text
                    except Exception:
                        phase = None

            # fallback heuristic if percent is missing
            if percent is None:
                now = time.time()
                last = st.session_state.get('last_progress_time', 0.0)
                if now - last > 0.5:
                    inc = random.randint(5, 12)
                    st.session_state.progress = min(98, st.session_state.progress + inc)
                    st.session_state.last_progress_time = now
                percent = st.session_state.progress

            with st.spinner(f"Processing: {phase or 'working'}..."):
                st.progress(percent)
                if phase:
                    st.write(f"Phase: {phase}")
                st.write("Processing in background. This page will auto-refresh until results are ready.")
        else:
            st.info("No result yet. Click Refresh Result to check or wait for processing to finish.")

