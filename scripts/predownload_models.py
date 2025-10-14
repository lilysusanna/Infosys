"""Predownload large ML models used by the app.

Usage examples:
  # from project root, inside venv
  python scripts/predownload_models.py --pyannote --whisper tiny --summarization facebook/bart-large-cnn

This script will download and cache models so Streamlit startup is faster.
"""
import os
import argparse


def predownload_pyannote(pipeline_id: str, token: str | None):
    from pyannote.audio import Pipeline
    print(f"Downloading pyannote pipeline {pipeline_id} ...")
    if not token:
        raise RuntimeError("Hugging Face token is required to download gated pyannote models. Set HUGGING_FACE_HUB_TOKEN in your environment.")
    Pipeline.from_pretrained(pipeline_id, token=token, revision="main")
    print("pyannote download finished")


def predownload_whisper(model_name: str):
    import whisper
    print(f"Downloading whisper model '{model_name}' ...")
    whisper.load_model(model_name)
    print("whisper download finished")


def predownload_summarization(model_name: str):
    try:
        from transformers import pipeline
    except Exception:
        print("transformers not installed; skipping summarization model download")
        return
    print(f"Downloading summarization model '{model_name}' ...")
    pipeline("summarization", model=model_name)
    print("summarization model download finished")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyannote", action="store_true", help="Prefetch pyannote speaker diarization pipeline")
    parser.add_argument("--pyannote-id", default="pyannote/speaker-diarization-3.1", help="pyannote pipeline id")
    parser.add_argument("--whisper", default=None, help="Whisper model name to predownload (tiny, base, small, medium, large)")
    parser.add_argument("--summarization", default=None, help="Hugging Face summarization model id (e.g. facebook/bart-large-cnn)")
    args = parser.parse_args()

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")

    if args.pyannote:
        try:
            predownload_pyannote(args.pyannote_id, token)
        except Exception as e:
            print("ERROR downloading pyannote:", e)
            return

    if args.whisper:
        try:
            predownload_whisper(args.whisper)
        except Exception as e:
            print("ERROR downloading whisper:", e)
            return

    if args.summarization:
        try:
            predownload_summarization(args.summarization)
        except Exception as e:
            print("ERROR downloading summarization model:", e)
            return

    print("All requested models downloaded / cached.")


if __name__ == '__main__':
    main()
