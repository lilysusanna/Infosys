AUDIO_DIR = "audio_wav"
OUTPUT_DIR = "results"
DIARIZED_DIR = f"{OUTPUT_DIR}/transcripts"
SUMMARIES_DIR = f"{OUTPUT_DIR}/summaries"
DIARIZE_MODEL = "pyannote/speaker-diarization"
WHISPER_MODEL = "small"  # whisper model size: tiny, small, medium, large
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
USE_HF_TOKEN = True
