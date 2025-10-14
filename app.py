import streamlit as st
from dotenv import load_dotenv
import os, time, threading, io, smtplib, random, sounddevice as sd, soundfile as sf
from datetime import datetime
from email.message import EmailMessage
from pipeline import add_to_pipeline, get_result

# --- Load environment variables once ---
load_dotenv(override=True)

# --- Streamlit Config ---
st.set_page_config(page_title="AI Meeting Summarizer", layout="wide")
st.title("AI Live Meeting Summarizer ⚡")

# --- Constants ---
FS = 16000
CHANNELS = 1
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# --- Initialize session state ---
for key, val in {
    "recording": False,
    "status": "idle",
    "current_file": None,
    "progress": 0,
    "last_result": None
}.items():
    st.session_state.setdefault(key, val)

# --- Recording Functions ---
def _record_thread(file_path):
    st.session_state.status = "recording"
    with sf.SoundFile(file_path, mode="w", samplerate=FS, channels=CHANNELS, subtype="PCM_16") as f:
        def callback(indata, frames, time_info, status):
            if not st.session_state.recording:
                raise sd.CallbackStop()
            f.write(indata)
        with sd.InputStream(samplerate=FS, channels=CHANNELS, callback=callback):
            while st.session_state.recording:
                time.sleep(0.1)
    st.session_state.status = "processing"
    add_to_pipeline(file_path)

def start_recording():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RECORDINGS_DIR, f"meeting_{timestamp}.wav")
    st.session_state.current_file = path
    st.session_state.recording = True
    threading.Thread(target=_record_thread, args=(path,), daemon=True).start()

def stop_recording():
    st.session_state.recording = False

# --- Email Function ---
def send_email_auto(recipient, subject, body):
    try:
        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        username = os.environ.get("SMTP_USERNAME")
        password = os.environ.get("SMTP_PASSWORD")
        msg = EmailMessage()
        msg["From"] = username
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.set_content(body)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

# --- Left Controls ---
left, right = st.columns([2, 3])

with left:
    st.subheader("🎙️ Record or Upload")
    uploaded = st.file_uploader("Upload a WAV/MP3 file", type=["wav", "mp3", "flac"])
    if uploaded:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECORDINGS_DIR, f"upload_{timestamp}_{uploaded.name}")
        with open(path, "wb") as f:
            f.write(uploaded.read())
        st.session_state.current_file = path
        st.session_state.status = "processing"
        add_to_pipeline(path)
        st.success("✅ File uploaded and processing started!")

    if not st.session_state.recording:
        if st.button("Start Recording 🎤"):
            start_recording()
    else:
        if st.button("Stop Recording ⏹️"):
            stop_recording()
            st.success("Recording stopped and processing started!")

    st.write(f"**Status:** {st.session_state.status}")

# --- Right Panel: Results ---
with right:
    st.subheader("📄 Live Results")

    # Auto-refresh every 3 seconds (without rerun)
    st_autorefresh = st.empty()
    result_placeholder = st.empty()

    if st.session_state.status == "processing":
        with st.spinner("Processing in background..."):
            time.sleep(1)
            result = get_result()
            if result:
                st.session_state.status = "done"
                st.session_state.last_result = result

    if st.session_state.status == "done" and st.session_state.last_result:
        transcript, diarized, summary = st.session_state.last_result
        tabs = st.tabs(["Transcript", "Diarized", "Summary", "Export", "Email"])

        with tabs[0]:
            st.text_area("Transcript", transcript, height=200)
        with tabs[1]:
            st.text_area("Diarized", "\n".join(diarized) if isinstance(diarized, list) else str(diarized), height=200)
        with tabs[2]:
            st.text_area("Summary", summary, height=200)
        with tabs[3]:
            # Export buttons: Markdown, TXT, Diarized TXT, and PDF (best-effort)
            md = f"# Meeting Summary\n\n## Summary\n{summary}\n\n## Transcript\n{transcript}\n"
            st.download_button("Download Summary (MD)", md, file_name="meeting_summary.md")
            st.download_button("Download Transcript (TXT)", transcript, file_name="meeting_transcript.txt")
            diarized_text = "\n".join(diarized) if isinstance(diarized, list) else str(diarized)
            st.download_button("Download Diarized (TXT)", diarized_text, file_name="meeting_diarized.txt")

            # Optional PDF export using reportlab if available
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                import io as _io

                def _make_pdf_bytes(title, summary_text, transcript_text):
                    buf = _io.BytesIO()
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
                            c.setFont("Helvetica", 10)
                        c.drawString(40, y, line)
                        y -= 14
                    y -= 10
                    for line in ("Transcript:",) + tuple(transcript_text.splitlines()):
                        if y < 60:
                            c.showPage()
                            y = height - 40
                            c.setFont("Helvetica", 10)
                        c.drawString(40, y, line)
                        y -= 12
                    c.save()
                    buf.seek(0)
                    return buf.read()

                pdf_bytes = _make_pdf_bytes("Meeting Summary", summary, transcript)
                st.download_button("Download PDF", pdf_bytes, file_name="meeting_summary.pdf", mime="application/pdf")
            except Exception as _pdf_err:
                st.caption(f"PDF export unavailable: {_pdf_err}")

        with tabs[4]:
            recipient = os.environ.get("EMAIL_RECIPIENT")
            if st.button("📧 Send Email Now"):
                if not recipient:
                    st.error("EMAIL_RECIPIENT not set in .env")
                else:
                    sent = send_email_auto(recipient, f"Meeting Summary - {datetime.now():%Y-%m-%d %H:%M}", summary)
                    if sent:
                        st.success(f"✅ Email sent successfully to {recipient}")
    elif st.session_state.status == "processing":
        st.progress(random.randint(10, 95))
    else:
        st.info("Upload or record to begin processing.")
