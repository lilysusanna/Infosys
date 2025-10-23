import streamlit as st
import whisper
from transformers import pipeline
from pyngrok import ngrok
import torch
import os

st.set_page_config(page_title="Smart Meeting Summarizer", page_icon="🎙")
st.title("🎙 Smart Meeting Summarizer")
st.write("Upload your meeting audio and get a summarized transcript.")

# --- Hugging Face Token (remove from code for security) ---
HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in your environment, not in code

# --- Whisper model for transcription ---
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

# --- Summarization pipeline ---
@st.cache_resource
def load_summarizer():
    # Make sure your Hugging Face token is used via environment variable
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        use_auth_token=HF_TOKEN
    )

summarizer = load_summarizer()

# --- File upload ---
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a"])
if uploaded_file:
    # Save the uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path)

    # --- Transcription ---
    st.write("Transcribing audio...")
    result = model.transcribe(file_path)
    transcript = result["text"]

    st.subheader("Transcript")
    st.write(transcript)

    # --- Summarization ---
    st.write("Generating summary...")
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

    st.subheader("Summary")
    st.write(summary)

    # --- Download option ---
    st.download_button(
        label="Download Transcript & Summary",
        data=f"Transcript:\n{transcript}\n\nSummary:\n{summary}",
        file_name="meeting_summary.txt",
        mime="text/plain"
    )
