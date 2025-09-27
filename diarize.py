
import os
os.environ["SPEECHBRAIN_LOCAL_STRATEGY"] = "copy"  # Windows fix: copy instead of symlink
os.environ["PYANNOTE_LOCAL_STRATEGY"] = "copy"
from pyannote.audio import Pipeline
import argparse

# --- CONFIG ---
DIARIZE_MODEL = "pyannote/speaker-diarization"  # PyAnnote diarization pipeline
USE_HF_TOKEN = "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"          # Replace with your Hugging Face token

# --- UTILS ---
def ensure_dirs(path):
    """Ensure the directory exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_text(file_path, text):
    ensure_dirs(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

# --- DIARIZATION FUNCTION ---
def diarize_audio(audio_path, out_rttm=None, out_simple=None):
    try:
        print("Loading diarization pipeline...")
        pipeline = Pipeline.from_pretrained(DIARIZE_MODEL, use_auth_token=USE_HF_TOKEN)
    except Exception as e:
        raise RuntimeError(
            "Failed to load pipeline. Make sure your Hugging Face token is correct "
            "and all gated models are accepted.\nOriginal error: " + str(e)
        )

    print(f"Diarizing audio: {audio_path} ...")
    diarization = pipeline(audio_path)

    # Set default output paths
    if out_rttm is None:
        out_rttm = os.path.join("results/diarization", os.path.basename(audio_path) + ".rttm")
    if out_simple is None:
        out_simple = os.path.join("results/transcripts", os.path.basename(audio_path) + ".diarized.txt")

    rttm_lines, simple_lines = [], []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = turn.start, turn.end
        rttm_lines.append(f"SPEAKER {os.path.basename(audio_path)} 1 {start:.3f} {end-start:.3f} <NA> <NA> {speaker} <NA>\n")
        simple_lines.append(f"[{speaker}] {start:.1f}-{end:.1f}")

    # Write outputs
    write_text(out_rttm, "".join(rttm_lines))
    write_text(out_simple, "\n".join(simple_lines))

    print(f"Diarization saved:\n  RTTM: {out_rttm}\n  Simple: {out_simple}")
    return out_rttm, out_simple

# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyAnnote Audio Speaker Diarization (Windows-friendly)")
    parser.add_argument("audio", help="Path to input audio file")
    args = parser.parse_args()
    diarize_audio(args.audio)
