-Live Meeting Summarizer Application 

The Live Meeting Summarizer Application is a powerful real-time meeting analysis tool that transcribes and summarizes audio discussions using state-of-the-art AI models. It provides a user-friendly interface built with Gradio, allowing users to upload audio recordings or capture live meeting streams for transcription, summarization, and evaluation.



-Overview

This application processes meeting audio through a complete AI pipeline:

Pipeline:
Audio → Transcription → Summarization (T5 & BART) → Evaluation (WER + ROUGE)

The tool leverages modern NLP and speech recognition technologies to convert meeting audio into accurate transcripts and concise summaries—perfect for extracting key points, action items, and decisions from discussions.
<img width="1902" height="778" alt="image" src="https://github.com/user-attachments/assets/d71d5048-f9db-40d1-85ca-85d86c40a4fe" />



 -Features

Audio-to-Text Conversion: Converts uploaded meeting recordings into accurate transcripts using automatic speech recognition (ASR) models.

AI-Powered Summarization: Summarizes transcriptions using T5 and BART models for concise and meaningful meeting summaries.

Evaluation Metrics: Automatically computes Word Error Rate (WER) and ROUGE scores for performance evaluation.

Language Support: Capable of handling multilingual audio with optional translation to English.

Gradio Web Interface: Provides an intuitive drag-and-drop UI for uploading .wav audio files.

Downloadable Output: Allows users to download transcripts and summaries in text format.




-Technology Stack

Frontend/UI: Gradio

Backend Processing: Python

ASR Engine: Whisper / wav2vec2 (for audio-to-text conversion)

Summarization Models: T5 and BART

Evaluation Metrics: WER and ROUGE

Audio Handling: FFmpeg



-Installation

Follow the steps below to set up and run the application:

Step 1: Clone the Repository
git clone https://github.com/Akhilesh653/Infosys/tree/main?tab=readme-ov-file
cd live-meeting-summarizer

Step 2: Set Up the Environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Run the Application
python app.py



-License

This project is licensed under the MIT License.



-Acknowledgements

Whisper
 — for accurate speech-to-text transcription

Gradio
 — for the interactive web interface

Hugging Face Transformers
— for T5 & BART models

Evaluate
 — for WER and ROUGE evaluation


