import whisper, argparse, json
from config import WHISPER_MODEL
from utils import write_text

def transcribe_whisper(audio_path, model_name=WHISPER_MODEL, out_json=None):
    model = whisper.load_model(model_name)
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
