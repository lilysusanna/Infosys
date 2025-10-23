🧠 Project Title: AI Meeting Summarizer

📋 Overview

The AI Meeting Summarizer is a machine learning–based web application that automatically transcribes and summarizes meeting audio recordings into concise, readable text summaries.

It helps users quickly review discussions, decisions, and action points from virtual meetings.

---
⚙ Features

🎤 Speech-to-Text Conversion using Vosk / Whisper models

🧾 Automatic Text Summarization using NLP-based techniques

🧠 Evaluation Metrics – WER, CER, ROUGE, and JIWER

💻 Streamlit UI for an interactive and user-friendly interface

📁 Supports uploading of meeting audio files and returns instant summaries

---
🧩 Modules

1. Audio Input Module – Takes recorded or uploaded meeting audio.

2. Speech Recognition Module – Converts speech to text using Vosk / Whisper.

3. Summarization Module – Uses NLP algorithms to generate concise summaries.

4. Evaluation Module – Measures accuracy and quality of generated summaries.

5. Streamlit UI Module – Displays the application interface and results to the user.
---
🛠 Tech Stack

Python

Streamlit (Web Interface)

Vosk / Whisper (Speech Recognition)

NLTK / Transformers (Text Summarization)

JiWER, ROUGE Metrics (Evaluation)

---

📂 Project Files

meeting_summariser_project.ipynb – Core Colab notebook

summary_evaluation.txt – Evaluation results

streamlit_app.py – Streamlit web interface

UI_flow.mp4 – Streamlit UI workflow video

---

🚀 How to Run

1. Open the .ipynb file in Google Colab or Jupyter Notebook.

2. Run all cells to generate transcription and summary outputs.

3. To run the UI:
streamlit run streamlit_app.py

4. Upload an audio file → view transcription + summary on screen.

---

👩‍💻 Developed By
sairishikamora
Infosys Springboard Virtual Internship – AI Meeting Summarization (2025)

---

🪪 License

This project is licensed under the MIT License.

---

