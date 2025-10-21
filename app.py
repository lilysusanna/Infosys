# app.py (This is save to file name)
import os
import uuid
import queue
import threading
import streamlit as st
import logging
from pipeline import PipelineWorker


# Suppress warnings

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# Config to audio path

REC_DIR = "recordings"
os.makedirs(REC_DIR, exist_ok=True)

st.set_page_config(page_title="STT + Diarization + Summarization", layout="centered")
st.title("🎙️ STT + Diarization + Summarization — Demo")

st.sidebar.header("Settings")
hf_token = st.sidebar.text_input("Hugging Face token (for pyannote)", type="password")
whisper_model = st.sidebar.selectbox("Whisper model", ["small", "medium", "large"], index=0)
summarizer_model = st.sidebar.text_input("Summarizer model (HF)", value="facebook/bart-large-cnn")


# Queues & Worker

if "task_queue" not in st.session_state:
    st.session_state.task_queue = queue.Queue()
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if "worker" not in st.session_state:
    try:
        st.session_state.worker = PipelineWorker(
            hf_token=hf_token or None,
            whisper_model=whisper_model,
            summarizer_model=summarizer_model,
            device="cpu"
        )
    except Exception as e: 
        st.error(f"Failed to initialize pipeline: {e}")
        st.stop()

    def _worker_loop(task_q, result_q, worker):
        while True:
            task = task_q.get()
            if task is None:
                break
            audio_path = task
            try:
                result = worker.process(audio_path)
            except Exception as e:
                result = {"error": str(e)}
            result_q.put((audio_path, result))

    t = threading.Thread(
        target=_worker_loop,
        args=(st.session_state.task_queue, st.session_state.result_queue, st.session_state.worker),
        daemon=True,
    )
    t.start()
    st.session_state.worker_thread = t


# Streamlit  UI

st.markdown("---")
st.info("Upload audio, then click **Process** to run ASR → diarization → summary.")

uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "flac"])

if uploaded:
    uploaded.seek(0)
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Process (ASR → Diarization → Summary)"):
        # Save uploaded file using absolute path
        filename = os.path.join(os.getcwd(), REC_DIR, f"{uuid.uuid4().hex}")
        ext = uploaded.name.split(".")[-1].lower()
        filename_ext = f"{filename}.{ext}"
        with open(filename_ext, "wb") as f:
            f.write(audio_bytes)

        st.success(f"Saved: {filename_ext}")
        st.write("File exists?", os.path.exists(filename_ext))

        # Convert to WAV if needed
        wav_file = st.session_state.worker.ensure_wav(filename_ext)
        st.write("Processing WAV file:", wav_file)

        # Queue file for processing
        st.session_state.task_queue.put(wav_file)

        # Wait for processing
        with st.spinner("Processing audio — ASR + diarization + summarization..."):
            result = None
            while True:
                audio_path, result = st.session_state.result_queue.get()
                if audio_path == wav_file:
                    break

        # Display results
        if result is None:
            st.error("No result returned from worker.")
        elif result.get("error"):
            st.error("Processing error: " + result.get("error"))
        else:
            st.subheader("📝 Diarized Transcript")
            for seg in result["diarized"]:
                st.write(f"**{seg['speaker']}** [{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}")

            st.subheader("📌 Summary")
            #st.write(result["summary"])
            summary_text = result["summary"]
            st.write(summary_text)
            
            # ✅ Add download button
            st.download_button(
                label="⬇️ Download Summary",
                data=summary_text,
                file_name="summary.txt",
                mime="text/plain"
            )


      