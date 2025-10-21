## E2E Audio Pipeline Testing App (Streamlit)

Stages: Upload, Transcribe (Whisper), Diarize (pyannote), Summarize (Transformers), Export (Markdown/PDF), Email.

### Quickstart
1. Create venv and install deps
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\\.venv\\Scripts\\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

2. Install FFmpeg (required by pydub/librosa)
- Windows (Chocolatey): `choco install ffmpeg`
- Or download from `https://ffmpeg.org` and ensure `ffmpeg` is on PATH.

3. (Optional) Enable speaker diarization
- Get a Hugging Face token with access to `pyannote/speaker-diarization-3.1`
- Set `HUGGINGFACE_TOKEN` in `.env`.

4. (Optional) Configure SMTP for email
- Copy `.env.example` to `.env` and set SMTP variables.

5. Run the app
```bash
streamlit run app.py
```

### Notes
- STT uses `faster-whisper`. Model size can be set via `WHISPER_MODEL_SIZE` in `.env`.
- Summarization uses `transformers` (DistilBART). It will download weights on first run.
- Diarization is optional; if token/model unavailable, the app skips it gracefully.
