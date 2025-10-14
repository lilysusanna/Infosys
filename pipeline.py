import threading
import queue
import os
from stt_whisper import transcribe_whisper
from diarization_module import diarize_audio
from summarization_module import summarize_text

task_queue = queue.Queue()
result_queue = queue.Queue()


def process_audio(audio_file):
    """Process an audio file: transcribe, diarize, summarize."""
    # 1️⃣ Transcribe
    try:
        transcript_res = transcribe_whisper(audio_file, out_json=None)
        transcript_text = transcript_res.get("text", "") if isinstance(transcript_res, dict) else str(transcript_res)
    except Exception:
        transcript_text = "(transcription failed)"

    # 2️⃣ Diarize
    try:
        diarized = diarize_audio(audio_file)
        if not diarized:
            diarized = [f"[Speaker 1]: {transcript_text}"]  # fallback
    except Exception:
        diarized = [f"[Speaker 1]: {transcript_text}"]

    # 3️⃣ Summarize
    try:
        summary_input = "\n".join(diarized)
        summary = summarize_text(summary_input)
        if not summary.strip():
            summary = "(summary unavailable)"
    except Exception:
        summary = "(summary failed)"

    result_queue.put((transcript_text, diarized, summary))


def add_to_pipeline(audio_file):
    task_queue.put(audio_file)
    threading.Thread(target=lambda: process_audio(audio_file), daemon=True).start()


def get_result():
    if not result_queue.empty():
        return result_queue.get()
    return None
