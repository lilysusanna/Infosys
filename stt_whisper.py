import whisper
import argparse
import json
import os
import torch
from config import WHISPER_MODEL
from utils import write_text


# Global cache to avoid reloading Whisper model for every transcription
_WHISPER_MODEL_CACHE = {}

def _detect_device():
    # Allow explicit override via environment variable MODEL_DEVICE (e.g. 'cuda', 'cpu')
    dev = os.environ.get('MODEL_DEVICE')
    if dev:
        return dev
    # otherwise prefer CUDA if available
    try:
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'

def transcribe_whisper(audio_path, model_name=WHISPER_MODEL, out_json=None, device=None):
    device = device or _detect_device()
    # Cache key by (model_name, device)
    cache_key = (model_name, device)
    model = _WHISPER_MODEL_CACHE.get(cache_key)
    if model is None:
        # whisper.load_model accepts device string like 'cuda' or 'cpu'
        model = whisper.load_model(model_name, device=device)
        _WHISPER_MODEL_CACHE[cache_key] = model
    # prefer fp16 on GPU for speed
    # model.transcribe will handle device selection internally
    result = model.transcribe(audio_path)
    if out_json:
        write_text(out_json, json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    transcribe_whisper(args.audio, out_json=args.out)
