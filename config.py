AUDIO_DIR = "audio_wav"
OUTPUT_DIR = "results"
DIARIZED_DIR = f"{OUTPUT_DIR}/transcripts"
SUMMARIES_DIR = f"{OUTPUT_DIR}/summaries"
DIARIZE_MODEL = "pyannote/speaker-diarization"
WHISPER_MODEL = "base"  # faster: choose tiny/base for low latency
SUMMARIZER_MODEL = "t5-small"  # faster summarizer by default
USE_HF_TOKEN = True
