# pipeline.py(This save to file name)
import os
import torch
import whisper
import librosa
import soundfile as sf
from transformers import pipeline as hf_pipeline
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

try:
    from pyannote.audio import Pipeline as DiarizationPipeline
except ImportError:
    DiarizationPipeline = None


class PipelineWorker:
    def __init__(self, hf_token=None, whisper_model="small", device="cpu", summarizer_model="facebook/bart-large-cnn"):
        # Whisper model on CPU
        self.whisper = whisper.load_model(whisper_model, device=device)

        #  Pyannote diarization
        if hf_token and DiarizationPipeline is not None:
            try:
                self.diarizer = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization", use_auth_token=hf_token
                )
            except Exception as e:
                print(f"Warning: Pyannote init failed: {e}")
                self.diarizer = None
        else:
            self.diarizer = None

        # HuggingFace summarizer
        hf_device = 0 if torch.cuda.is_available() else -1
        self.summarizer = hf_pipeline("summarization", model=summarizer_model, device=hf_device)

    def ensure_wav(self, audio_path):
        """Convert mp3/m4a/flac to WAV (16kHz) if needed"""
        if audio_path.lower().endswith(".wav"):
            return os.path.abspath(audio_path)
        y, sr = librosa.load(audio_path, sr=16000)
        wav_path = os.path.abspath(audio_path + ".wav")
        sf.write(wav_path, y, sr)
        return wav_path

    def transcribe(self, audio_path):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self.whisper.transcribe(audio_path, verbose=False)

    def diarize(self, audio_path):
        if self.diarizer is None:
            return []
        diarization = self.diarizer(audio_path)
        diarized_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarized_segments.append({
                "speaker": speaker,
                "start": turn.start,
                "end": turn.end,
                "text": ""
            })
        return diarized_segments

    def align_transcript(self, transcript, diarized_segments):
        """Assign ASR segments to speakers"""
        if not diarized_segments:
            return [{"speaker": "Unknown", "start": seg["start"], "end": seg["end"], "text": seg["text"]}
                    for seg in transcript["segments"]]

        aligned = []
        for seg in transcript["segments"]:
            mid_time = (seg["start"] + seg["end"]) / 2
            speaker = "Unknown"
            for d in diarized_segments:
                if d["start"] <= mid_time <= d["end"]:
                    speaker = d["speaker"]
                    break
            aligned.append({
                "speaker": speaker,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"]
            })
        return aligned

    def summarize(self, text):
        if not text.strip():
            return ""
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summaries = [
            self.summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
            for chunk in chunks
        ]
        return " ".join(summaries)

    def process(self, audio_path):
        # 1. Convert to WAV
        audio_path = self.ensure_wav(audio_path)

        # 2. Transcribe
        transcript = self.transcribe(audio_path)

        # 3. Diarize
        diarized_segments = self.diarize(audio_path)

        # 4. Align transcript
        aligned = self.align_transcript(transcript, diarized_segments)

        # 5. Summarize
        full_text = " ".join(seg["text"] for seg in aligned)
        summary = self.summarize(full_text)

        return {
            "transcript": transcript,
            "diarized": aligned,
            "summary": summary
        }