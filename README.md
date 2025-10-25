# AI Meeting Summarizer 📝🤖

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/meeting_summarizer_project/blob/main/Ai_Meeting_Summarizer.ipynb)  
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)  

A Python-based AI project that **transcribes, summarizes, and analyzes meeting audio recordings**. It identifies speakers and generates concise summaries for efficient meeting documentation.

---

## 🔹 Demo
Below are screenshots showing the workflow of the notebook:

<table>
  <tr>
    <td><img src="screenshot1.png" width="250"></td>
    <td><img src="screenshot2.png" width="250"></td>
    <td><img src="screenshot3.png" width="250"></td>
  </tr>
  <tr>
    <td><img src="screenshot4.png" width="250"></td>
    <td><img src="screenshot5.png" width="250"></td>
    <td></td>
  </tr>
</table>

---

## 🔹 Features
- **Audio Transcription**: Convert meeting audio files (.mp3, .wav) into text  
- **Speaker Diarization**: Identify and label individual speakers  
- **Automatic Summarization**: Generate concise summaries of meetings  
- **ROUGE Evaluation**: Evaluate summary quality using ROUGE metrics  
- **Multi-file Support**: Upload multiple meeting recordings for batch processing  

---

## 🔹 Tech Stack
- **Python 3**  
- **Google Colab Notebook**  
- Libraries:
  - `speech_recognition`  
  - `transformers` (for summarization)  
  - `pyannote.audio` (for speaker diarization)  
  - `evaluate` (for ROUGE metrics)  

---

## 🔹 How to Use
1. Open `Ai_Meeting_Summarizer.ipynb` in [Google Colab](https://colab.research.google.com/).  
2. Upload your meeting audio file (.mp3 or .wav).  
3. Run the notebook cells to:
   - Generate transcription  
   - Identify speakers  
   - Produce a summary  
   - Evaluate the summary (optional)  
4. Download results as needed.  

---

## 🔹 Project Structure
Ai_Meeting_Summarizer/
│
├── Ai_Meeting_Summarizer.ipynb # Main notebook
├── README.md # Project documentation
│ ├── screenshot1.png
│ ├── screenshot2.png
│ ├── screenshot3.png
│ ├── screenshot4.png
│ └── screenshot5.png
└── requirements.txt (optional) # Python dependencies

---

## 🔹 Future Improvements
- Real-time transcription for live meetings  
- GUI or web interface (Streamlit / Flask)  
- Multi-language support for transcription & summarization  
- Export summaries in PDF or Word format  

---

## 🔹 Author
**Sairishika Mora**  
B.Tech IT | AI & NLP Enthusiast  

