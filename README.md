# Live Meeting Summarizer

## Overview
**Live Meeting Summarizer** is a real-time application that converts meeting audio into structured summaries. It captures live speech, separates speakers, and generates summaries using LLMs or Hugging Face transformers. Users can view live transcription, export summaries as Markdown/PDF, or send them via email.

---

## Features
- **Live Audio Capture & STT:** Record and transcribe in real time using Vosk or Whisper.
- **Audio File Upload:** Process pre-recorded audio files for transcription and summarization.
- **Speaker Diarization:** Identify and separate speakers with pyannote.audio.
- **LLM-Based Summarization:** Structured summaries via Groq LLaMA 3.1, T5, or BART.
- **Interactive UI:** Streamlit interface with Start/Stop recording, live transcription, and summary view.
- **Export & Email:** One-click download or email summaries.
- **Backend Pipeline:** Multi-threaded and asynchronous processing.

---

## Tech Stack
| Component | Tools / Libraries |
|-----------|------------------|
| STT | Vosk, Whisper, PyAudio, SoundDevice |
| Diarization | pyannote.audio, torchaudio |
| Summarization | LLaMA 3.1 (Groq API), T5, BART (Hugging Face) |
| Frontend | Streamlit |
| Backend | Python threading, asyncio, queue |
| Evaluation | jiwer (WER), rouge_score, BLEU |
| Export & Logging | JSON, Markdown, Parquet, smtplib |

---

## Usage
1. Install dependencies:
```bash
pip install -r requirements.txt
2. Run the Streamlit app:

streamlit run streamlit_app.py


3. Use the UI:

Start recording → live transcription updates

Stop recording → diarized transcript + summary

Upload Audio File → process existing recordings

Export/Email summary

Evaluation

WER (STT): < 15%

DER (Diarization): < 20%

ROUGE (Summary): > 0.4

UI: Smooth real-time interaction, no lag
