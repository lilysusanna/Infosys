import whisper
import os
import json
try:
    import torch  # optional but preferred for device detection
except Exception:
    torch = None

# Default model
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")  # e.g., "base", "small", "medium", "large"


def _detect_device():
    if os.environ.get("MODEL_DEVICE"):
        return os.environ.get("MODEL_DEVICE")
    try:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def transcribe_whisper(audio_path, model_name=WHISPER_MODEL, out_json=None, device=None):
    """Transcribe an audio file using Whisper and return a dict with 'text' and 'segments'."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device = device or _detect_device()
    # Load model with graceful fallbacks (CUDA -> CPU; model name fallback)
    try:
        model = whisper.load_model(model_name, device=device)
    except Exception:
        try:
            # fallback to CPU if GPU load failed
            model = whisper.load_model(model_name, device="cpu")
            device = "cpu"
        except Exception:
            # fallback to a smaller model on CPU
            fallback_name = "base" if model_name != "base" else "small"
            model = whisper.load_model(fallback_name, device="cpu")
            device = "cpu"

    forced_lang = os.environ.get("WHISPER_LANGUAGE")
    try:
        result = model.transcribe(
            audio_path,
            language=forced_lang if forced_lang else None,
            fp16=(device != "cpu"),
        )
    except Exception:
        # retry with defaults
        result = model.transcribe(audio_path)

    if "segments" not in result:
        result["segments"] = [{"start": 0, "end": 0, "text": result.get("text", "")}]

    if out_json:
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return result

if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python stt_whisper.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    result = transcribe_whisper(audio_file)
    print("Transcription:", result.get("text", ""))
    with open(audio_file.replace(".wav", "_stt.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
