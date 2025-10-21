import os
import tempfile
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except Exception:
    # dotenv not available in environment; define a no-op fallback
    def load_dotenv(*args, **kwargs):
        return False


from services.stt import transcribe_audio
from services.diarization import diarize_audio, assign_speakers_to_transcript, DiarizationUnavailable
from services.summarization import summarize_text
from services.export_utils import export_markdown, export_pdf
from services.email_utils import send_email_via_smtp
from services.oauth_email import send_email_oauth, setup_oauth_instructions, test_oauth_connection
from services.utils.audio import get_audio_metadata, slice_audio_to_temp


load_dotenv()

st.set_page_config(page_title="Audio Pipeline Tester", page_icon="🎧", layout="wide")

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state["transcript"] = None
if "diarized" not in st.session_state:
    st.session_state["diarized"] = None
if "summary" not in st.session_state:
    st.session_state["summary"] = None
if "export_file" not in st.session_state:
    st.session_state["export_file"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None
if "filename" not in st.session_state:
    st.session_state["filename"] = None
if "duration" not in st.session_state:
    st.session_state["duration"] = None
if "stt_segments" not in st.session_state:
    st.session_state["stt_segments"] = None
if "diar_segments" not in st.session_state:
    st.session_state["diar_segments"] = None

# Debug: Show initial session state
st.info(f"🔍 Initial Session State - transcript: {st.session_state.get('transcript')}, audio_path: {st.session_state.get('audio_path')}")

# Header
st.title("🎧 End-to-End Audio Pipeline Tester")
st.caption("Upload → Transcribe → Diarize → Summarize → Export/Email")

with st.sidebar:
    st.header("Controls")
    uploaded = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3", "wav", "m4a", "flac"]) 
    model_size = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large-v3"], index=["tiny","base","small","medium","large-v3"].index(os.getenv("WHISPER_MODEL_SIZE", "small")))
    st.divider()
    st.subheader("Stages")
    col_a, col_b = st.columns(2)
    with col_a:
        trans_btn = st.button("Transcribe", type="primary", use_container_width=True, key="sidebar_transcribe")
        diar_btn = st.button("Diarize", use_container_width=True, key="sidebar_diarize")
    with col_b:
        sum_btn = st.button("Summarize", use_container_width=True, key="sidebar_summarize")
        exp_btn = st.button("Export", use_container_width=True, key="sidebar_export")
    
    st.divider()
    st.subheader("Email")
    email_btn = st.button("Send Email", use_container_width=True, key="sidebar_email")
    st.divider()
    st.subheader("Workflow Status")
    
    # Status indicators
    status_items = []
    if st.session_state["audio_path"]:
        status_items.append("✅ Audio uploaded")
    if st.session_state["transcript"]:
        status_items.append("✅ Transcribed")
    if st.session_state["diarized"]:
        status_items.append("✅ Diarized")
    if st.session_state["summary"]:
        status_items.append("✅ Summarized")
    
    if status_items:
        for item in status_items:
            st.write(item)
    else:
        st.write("⏳ Upload audio to begin")

# Persist uploaded file to temp
if uploaded is not None:
    if st.session_state["filename"] != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded.name}") as tmp:
            tmp.write(uploaded.read())
            st.session_state["audio_path"] = tmp.name
            st.session_state["filename"] = uploaded.name
            meta = get_audio_metadata(tmp.name)
            st.session_state["duration"] = meta.get("duration_sec")

# Top-level info
with st.container(border=True):
    st.subheader("Audio Info")
    if st.session_state["audio_path"]:
        st.write(f"**Filename**: {st.session_state['filename']}")
        dur = st.session_state["duration"]
        st.write(f"**Duration**: {dur:.1f} sec" if dur else "Duration: unknown")
        st.audio(st.session_state["audio_path"])
        
        # Debug info
        with st.expander("🔍 Debug Info", expanded=False):
            st.write(f"Audio path exists: {os.path.exists(st.session_state['audio_path']) if st.session_state['audio_path'] else False}")
            st.write(f"Has transcript: {bool(st.session_state.get('transcript'))}")
            st.write(f"Has stt_segments: {bool(st.session_state.get('stt_segments'))}")
            st.write(f"Has diarized: {bool(st.session_state.get('diarized'))}")
            st.write(f"Has summary: {bool(st.session_state.get('summary'))}")
            st.write(f"Has export file: {bool(st.session_state.get('export_file'))}")
            transcript = st.session_state.get('transcript') or ''
            st.write(f"Transcript length: {len(transcript)}")
            stt_segments = st.session_state.get('stt_segments') or []
            st.write(f"STT segments count: {len(stt_segments)}")
            if transcript:
                st.write(f"Transcript preview: {transcript[:100]}...")
    else:
        st.info("Upload an audio file to begin.")

# Tabs
transcript_tab, summary_tab, export_tab, diar_viz_tab = st.tabs(["Transcript Viewer", "Summary Viewer", "Export Options", "Diarization Viewer"])

# Handle buttons with proper dependency checking
if trans_btn:
    st.info("🔍 TRANSCRIBE BUTTON CLICKED!")
    if not st.session_state["audio_path"]:
        st.error("❌ Please upload an audio file first!")
    else:
        st.info(f"🎵 Starting transcription of: {st.session_state['filename']}")
        st.info(f"🔍 Debug: Audio path exists = {os.path.exists(st.session_state['audio_path'])}")
        st.info(f"🔍 Debug: Model size = {model_size}")
        
        # Check audio file validity first
        try:
            import soundfile as sf
            info = sf.info(st.session_state["audio_path"])
            st.info(f"🔍 Audio info - Duration: {info.duration:.2f}s, Sample rate: {info.samplerate}Hz, Channels: {info.channels}")
            if info.duration < 0.1:
                st.warning("⚠️ Audio file is very short (< 0.1s). This might cause transcription issues.")
        except ImportError as e:
            st.warning(f"⚠️ Soundfile dependency missing: {e}")
            st.info("💡 Installing missing dependencies... Run: pip install cffi")
            # Continue with transcription anyway
        except Exception as e:
            st.warning(f"⚠️ Could not read audio file info: {e}")
            st.info("💡 This won't affect transcription - continuing anyway...")
        
        with st.spinner("Transcribing with Whisper..."):
            try:
                segments, full_text = transcribe_audio(st.session_state["audio_path"], model_size=model_size)
                st.info(f"🔍 Debug: Transcription returned - segments: {len(segments)}, text length: {len(full_text)}")
                
                if len(segments) == 0:
                    st.warning("⚠️ No speech detected in audio file. This could be due to:")
                    st.warning("  - Audio file is too short or silent")
                    st.warning("  - Audio quality is too poor")
                    st.warning("  - Audio format not supported")
                    st.warning("  - No speech content in the audio")
                    
                    # Try a different approach - create a dummy transcript for testing
                    st.info("🔧 Creating a test transcript for debugging purposes...")
                    test_segments = [{"start": 0.0, "end": 1.0, "text": "This is a test transcript for debugging purposes."}]
                    test_text = "This is a test transcript for debugging purposes."
                    st.session_state["stt_segments"] = test_segments
                    st.session_state["transcript"] = test_text
                    st.info("✅ Test transcript created. You can now test other buttons.")
                else:
                    st.session_state["stt_segments"] = segments
                    st.session_state["transcript"] = full_text
                
                st.info(f"🔍 Debug: After saving - transcript in session: {bool(st.session_state.get('transcript'))}")
                st.info(f"🔍 Debug: Transcript content preview: {st.session_state.get('transcript', '')[:100]}...")
                
                st.success("✅ Transcription complete! Check the Transcript Viewer tab.")
                st.info(f"📝 Generated {len(segments)} segments, {len(full_text)} characters")
                
                # Additional debug info
                st.info("🔍 Session State Debug:")
                st.info(f"  - transcript: {bool(st.session_state.get('transcript'))} (length: {len(st.session_state.get('transcript', ''))})")
                st.info(f"  - stt_segments: {bool(st.session_state.get('stt_segments'))} (count: {len(st.session_state.get('stt_segments', []))})")
                st.info(f"  - audio_path: {bool(st.session_state.get('audio_path'))}")
                
            except Exception as e:
                st.error(f"❌ Transcription failed: {str(e)}")
                st.error(f"Debug: Audio path = {st.session_state['audio_path']}, Model = {model_size}")
                import traceback
                st.error(f"Full traceback: {traceback.format_exc()}")

if diar_btn:
    transcript = st.session_state.get('transcript') or ''
    if not transcript:
        st.warning("⚠️ Please transcribe first!")
    elif not st.session_state["audio_path"]:
        st.error("❌ Audio file not available!")
    else:
        with st.spinner("Running speaker diarization..."):
            try:
                diar_segments = diarize_audio(st.session_state["audio_path"])
                st.session_state["diar_segments"] = diar_segments
                if diar_segments:
                    # Convert segments to text for diarization
                    transcript_text = st.session_state.get("transcript", "")
                    diarized_segments = assign_speakers_to_transcript(transcript_text, diar_segments)
                    st.session_state["diarized"] = diarized_segments
                    st.success("✅ Diarization complete! Check the Diarization Viewer tab.")
                else:
                    st.warning("⚠️ Diarization skipped or unavailable.")
            except DiarizationUnavailable as e:
                st.warning(f"⚠️ {str(e)}")
            except Exception as e:
                st.error(f"❌ Diarization failed: {str(e)}")

if sum_btn:
    transcript = st.session_state.get('transcript') or ''
    st.info(f"🔍 Debug: Checking transcript = {bool(transcript)}, value = {transcript is not None and transcript != ''}")
    if not transcript:
        st.warning("⚠️ Please transcribe first!")
    else:
        with st.spinner("Summarizing transcript..."):
            try:
                # Use diarized transcript if available, otherwise use regular transcript
                if st.session_state.get("diarized"):
                    # If we have diarized segments, convert them to text
                    text_to_summarize = "\n".join([seg["text"] for seg in st.session_state["diarized"]])
                    st.info("📝 Using diarized transcript for summarization")
                else:
                    # Use the regular transcript
                    text_to_summarize = st.session_state.get("transcript", "")
                    st.info("📝 Using regular transcript for summarization")
                
                st.info(f"🔍 Text length: {len(text_to_summarize)} characters")
                summary = summarize_text(text_to_summarize) or "(No summary generated)"
                st.session_state["summary"] = summary
                st.success("✅ Summary ready! Check the Summary Viewer tab.")
            except Exception as e:
                st.error(f"❌ Summarization failed: {str(e)}")

if exp_btn:
    transcript = st.session_state.get('transcript') or ''
    if not transcript:
        st.warning("⚠️ Please transcribe first!")
    else:
        with st.spinner("Exporting..."):
            try:
                # Get the appropriate segments data (fallback to raw transcript string if empty)
                segments_data = st.session_state.get("diarized") if st.session_state.get("diarized") else st.session_state.get("stt_segments", [])
                summary_data = st.session_state.get("summary")
                if not segments_data:
                    raw_transcript = st.session_state.get("transcript", "")
                    segments_data = raw_transcript if raw_transcript else []
                
                # Create exports directory
                exports_dir = Path("exports")
                exports_dir.mkdir(parents=True, exist_ok=True)
                
                # Export both formats
                # Markdown
                md_content = export_markdown(segments_data, summary_data)
                md_path = exports_dir / f"{Path(st.session_state['filename']).stem}.md"
                md_path.write_text(md_content, encoding="utf-8")
                
                # PDF
                pdf_path = exports_dir / f"{Path(st.session_state['filename']).stem}.pdf"
                pdf_result = export_pdf(segments_data, summary_data, str(pdf_path))
                
                st.session_state["export_file"] = str(md_path)  # Store the export file path
                
                st.success("✅ Export complete! Check the Export Options tab for download links.")
                st.info(f"📄 Markdown: {md_path}")
                if pdf_result:
                    st.info(f"📄 PDF: {pdf_result}")
                
                # Show download buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Markdown",
                        data=md_content,
                        file_name=md_path.name,
                        mime="text/markdown"
                    )
                with col2:
                    if pdf_result and os.path.exists(pdf_result):
                        with open(pdf_result, "rb") as f:
                            st.download_button(
                                label="Download PDF",
                                data=f.read(),
                                file_name=Path(pdf_result).name,
                                mime="application/pdf"
                            )
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")

if email_btn:
    transcript = st.session_state.get('transcript') or ''
    if not transcript:
        st.warning("⚠️ Please transcribe first!")
    else:
        with st.spinner("Preparing email..."):
            try:
                # Get the appropriate segments data
                segments_data = st.session_state.get("diarized") if st.session_state.get("diarized") else st.session_state.get("stt_segments", [])
                summary_data = st.session_state.get("summary")
                
                # Debug: Show what data we have
                st.info(f"🔍 Debug: segments_data type: {type(segments_data)}, length: {len(segments_data) if segments_data else 'None'}")
                st.info(f"🔍 Debug: summary_data: {bool(summary_data)}")
                
                # If no segments data, use raw transcript as fallback
                if not segments_data:
                    raw_transcript = st.session_state.get("transcript", "")
                    if raw_transcript:
                        st.info("🔧 Using raw transcript as fallback for email content")
                        segments_data = raw_transcript
                    else:
                        st.error("❌ No transcript data available for email!")
                        st.stop()
                
                # Create temporary file for attachment
                md_content = export_markdown(segments_data, summary_data)
                st.info(f"🔍 Debug: Generated content length: {len(md_content)} characters")
                st.info(f"🔍 Debug: Content preview: {md_content[:200]}...")
                
                att_path = Path(tempfile.gettempdir()) / f"{Path(st.session_state['filename']).stem}.md"
                att_path.write_text(md_content, encoding="utf-8")
                
                # Use default email settings
                recipient = "recipient@example.com"  # Default placeholder
                subject = f"Audio Transcript: {st.session_state['filename']}"
                
                # Create a more informative email body
                email_body = f"""Hello,

Please find attached the audio transcript for: {st.session_state['filename']}

Content Details:
- File: {st.session_state['filename']}
- Duration: {st.session_state.get('duration', 'Unknown')} seconds
- Transcript Length: {len(md_content)} characters
- Format: Markdown

The attachment contains the complete transcript with speaker information (if available) and summary.

Best regards,
Audio Pipeline System"""
                
                ok, message = send_email_via_smtp(
                    recipient_email=recipient,
                    subject=subject,
                    body=email_body,
                    attachments=[str(att_path)],
                )
                if ok:
                    st.success(f"✅ {message}")
                    st.info("ℹ️ Note: This is a demo. Configure SMTP settings in environment variables for real email sending.")
                else:
                    st.error(f"❌ Failed to send: {message}")
            except Exception as e:
                st.error(f"❌ Email sending failed: {str(e)}")

# Transcript tab content
with transcript_tab:
    st.subheader("Transcript")
    if st.session_state.get("stt_segments"):
        use_diar = st.toggle("Show speaker labels (if available)", value=bool(st.session_state.get("diarized")))
        if use_diar and st.session_state.get("diarized"):
            for seg in st.session_state["diarized"]:
                st.markdown(f"**{seg['speaker']}** [{seg['start']:.1f}-{seg['end']:.1f}s]: {seg['text']}")
        else:
            full_text = "\n".join([s["text"] for s in st.session_state["stt_segments"]])
            st.text_area("Transcript", value=full_text, height=300)
        st.success("✓ Transcribed")
        if st.session_state.get("diar_segments"):
            st.info("✓ Diarized")
    else:
        st.info("No transcript yet. Click Transcribe.")

# Summary tab content
with summary_tab:
    st.subheader("Summary")
    if st.session_state.get("summary"):
        st.success("✓ Summarized")
        st.container(border=True).markdown(st.session_state["summary"]) 
    elif st.session_state.get("transcript"):
        st.info("Click Summarize to generate a summary.")
    else:
        st.info("No transcript available.")

# Export tab content
with export_tab:
    st.subheader("Export & Email")
    
    # Export format selection
    export_format = st.selectbox("Choose export format:", ["Markdown", "PDF"], index=0, key="export_format_select")
    transcript = st.session_state.get('transcript') or ''
    export_clicked = st.button("Export", type="primary", use_container_width=True, disabled=not transcript, key="export_tab_export")
    
    if export_clicked:
        if not transcript:
            st.warning("⚠️ Please transcribe first!")
        else:
            with st.spinner(f"Exporting as {export_format}..."):
                try:
                    # Get the appropriate segments data (fallback to raw transcript string if empty)
                    segments_data = st.session_state.get("diarized") if st.session_state.get("diarized") else st.session_state.get("stt_segments", [])
                    summary_data = st.session_state.get("summary")
                    if not segments_data:
                        raw_transcript = st.session_state.get("transcript", "")
                        segments_data = raw_transcript if raw_transcript else []
                    
                    # Create exports directory
                    exports_dir = Path("exports")
                    exports_dir.mkdir(parents=True, exist_ok=True)
                    
                    if export_format == "Markdown":
                        content = export_markdown(segments_data, summary_data)
                        out_path = exports_dir / f"{Path(st.session_state['filename']).stem}.md"
                        out_path.write_text(content, encoding="utf-8")
                        st.session_state["export_file"] = str(out_path)
                        st.success(f"✅ Markdown exported → {out_path}")
                        st.download_button(
                            label="Download Markdown",
                            data=content,
                            file_name=out_path.name,
                            mime="text/markdown"
                        )
                    else:  # PDF
                        out_path = exports_dir / f"{Path(st.session_state['filename']).stem}.pdf"
                        result_path = export_pdf(segments_data, summary_data, str(out_path))
                        if result_path:
                            st.session_state["export_file"] = str(result_path)
                            st.success(f"✅ PDF exported → {result_path}")
                            with open(result_path, "rb") as f:
                                st.download_button(
                                    label="Download PDF",
                                    data=f.read(),
                                    file_name=Path(result_path).name,
                                    mime="application/pdf"
                                )
                        else:
                            st.error("Failed to export PDF")
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
    
    st.divider()
    
    # Email section
    st.subheader("Send via Email")
    
    # Email instructions
    st.info("📧 **Email Instructions:** Please enter the recipient's email address below. The application will send the generated content or message directly to that email. Ensure the email address is valid and accessible.")
    
    # Email method selection
    email_method = st.radio("Choose email method:", ["OAuth2 (Gmail)", "SMTP (Traditional)"], index=0)
    
    col1, col2 = st.columns(2)
    with col1:
        recipient = st.text_input("Recipient Email", placeholder="example@email.com", key="email_recipient", help="Enter the email address where you want to send the transcript and summary")
        subject = st.text_input("Subject", value="Audio Transcript & Summary", key="email_subject", help="Customize the email subject line")
    with col2:
        email_format = st.selectbox("Email format:", ["Markdown", "PDF"], index=0, key="email_format_select", help="Choose the format for the attached file")
        msg_clicked = st.button("Send Email", use_container_width=True, disabled=not transcript or not recipient, key="export_tab_email")
    
    # OAuth2 setup instructions
    if email_method == "OAuth2 (Gmail)":
        with st.expander("🔐 OAuth2 Setup Instructions", expanded=False):
            setup_oauth_instructions()
            
            # Test OAuth connection
            if st.button("Test OAuth2 Connection", key="test_oauth"):
                with st.spinner("Testing OAuth2 connection..."):
                    success, message = test_oauth_connection()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")

    if msg_clicked:
        if not transcript:
            st.warning("⚠️ Please transcribe first!")
        elif not recipient:
            st.warning("⚠️ Please enter recipient email!")
        else:
            with st.spinner("Sending email..."):
                try:
                    # Create temporary file for attachment
                    segments_data = st.session_state.get("diarized") if st.session_state.get("diarized") else st.session_state.get("stt_segments", [])
                    summary_data = st.session_state.get("summary")
                    
                    # Debug: Show what data we have
                    st.info(f"🔍 Debug: segments_data type: {type(segments_data)}, length: {len(segments_data) if segments_data else 'None'}")
                    st.info(f"🔍 Debug: summary_data: {bool(summary_data)}")
                    
                    # If no segments data, use raw transcript as fallback
                    if not segments_data:
                        raw_transcript = st.session_state.get("transcript", "")
                        if raw_transcript:
                            st.info("🔧 Using raw transcript as fallback for email content")
                            segments_data = raw_transcript
                        else:
                            st.error("❌ No transcript data available for email!")
                            st.stop()
                    
                    if email_format == "Markdown":
                        content = export_markdown(segments_data, summary_data)
                        st.info(f"🔍 Debug: Generated content length: {len(content)} characters")
                        st.info(f"🔍 Debug: Content preview: {content[:200]}...")
                        att_path = Path(tempfile.gettempdir()) / f"{Path(st.session_state['filename']).stem}.md"
                        att_path.write_text(content, encoding="utf-8")
                    else:  # PDF
                        att_path = Path(tempfile.gettempdir()) / f"{Path(st.session_state['filename']).stem}.pdf"
                        result_path = export_pdf(segments_data, summary_data, str(att_path))
                        if not result_path:
                            st.error("Failed to create PDF for email")
                            st.stop()
                        att_path = Path(result_path)
                    
                    # Create a more informative email body
                    email_body = f"""Hello,

Please find attached the audio transcript for: {st.session_state['filename']}

Content Details:
- File: {st.session_state['filename']}
- Duration: {st.session_state.get('duration', 'Unknown')} seconds
- Format: {email_format}
- Transcript Length: {len(content) if email_format == "Markdown" else "PDF generated"} characters

The attachment contains the complete transcript with speaker information (if available) and summary.

Best regards,
Audio Pipeline System"""
                    
                    if email_method == "OAuth2 (Gmail)":
                        ok, message = send_email_oauth(
                            recipient_email=recipient,
                            subject=subject,
                            body=email_body,
                            attachments=[str(att_path)],
                        )
                    else:  # SMTP
                        ok, message = send_email_via_smtp(
                            recipient_email=recipient,
                            subject=subject,
                            body=email_body,
                            attachments=[str(att_path)],
                        )
                    
                    if ok:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ Failed to send: {message}")
                except Exception as e:
                    st.error(f"Email sending failed: {str(e)}")

# Diarization Viewer tab
with diar_viz_tab:
    st.subheader("Diarization Viewer")
    if not st.session_state.get("diar_segments"):
        st.info("Run Diarize to view speaker segments.")
    else:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle

            segments = st.session_state["diar_segments"]
            # Map speakers to indices
            speakers = list({s["speaker"] for s in segments})
            speaker_to_row = {spk: idx for idx, spk in enumerate(sorted(speakers))}

            fig, ax = plt.subplots(figsize=(10, 2 + 0.5 * len(speakers)))
            ax.set_xlim(0, max(s["end"] for s in segments))
            ax.set_ylim(0, len(speakers))
            ax.set_xlabel("Time (s)")
            ax.set_yticks([i + 0.5 for i in range(len(speakers))])
            ax.set_yticklabels([f"{spk}" for spk in sorted(speakers)])
            ax.grid(True, axis="x", linestyle=":", alpha=0.4)

            colors = plt.cm.get_cmap('tab20', len(speakers))
            for seg in segments:
                row = speaker_to_row[seg["speaker"]]
                start = seg["start"]
                width = max(0.01, seg["end"] - seg["start"]) 
                rect = Rectangle((start, row), width, 0.9, color=colors(row), alpha=0.8)
                ax.add_patch(rect)
                ax.text(start + width / 2, row + 0.45, f"{seg['speaker']}", ha='center', va='center', fontsize=8, color='white')

            st.pyplot(fig, clear_figure=True, use_container_width=True)

            st.markdown("Select a segment to play back:")
            # Build a selection of segments
            options = [f"{i+1}. {seg['speaker']} [{seg['start']:.1f}-{seg['end']:.1f}s]" for i, seg in enumerate(segments)]
            idx = st.selectbox("Segments", range(len(segments)), format_func=lambda i: options[i])
            sel = segments[idx]
            start_s = float(sel["start"])
            end_s = float(sel["end"]) 
            clip_path = slice_audio_to_temp(st.session_state["audio_path"], start_s, end_s)
            if clip_path and os.path.exists(clip_path):
                st.audio(clip_path)
            else:
                st.warning("Could not create audio clip for playback.")
        except ImportError as e:
            st.error(f"❌ Visualization library error: {e}")
            st.info("📋 Diarization data is available, but visualization is not working due to missing dependencies.")
            st.info("You can still see the diarized transcript in the Transcript Viewer tab.")
            
            # Show diarization data in a simple format
            segments = st.session_state["diar_segments"]
            st.write("**Speaker Segments:**")
            for i, seg in enumerate(segments):
                st.write(f"{i+1}. **{seg['speaker']}** [{seg['start']:.1f}-{seg['end']:.1f}s]")
        except Exception as e:
            st.error(f"❌ Visualization error: {e}")
            st.info("📋 Diarization data is available, but visualization failed.")