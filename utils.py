import os
from pathlib import Path

def ensure_dirs():
    Path("results/diarization").mkdir(parents=True, exist_ok=True)
    Path("results/transcripts").mkdir(parents=True, exist_ok=True)
    Path("results/summaries").mkdir(parents=True, exist_ok=True)

def write_text(path, text):
    ensure_dirs()
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
