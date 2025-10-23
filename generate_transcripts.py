import os
import time
from glob import glob
from jiwer import wer
import whisper
from vosk import Model, KaldiRecognizer
import wave
import json
import re
from deepmultilingualpunctuation import PunctuationModel

# ===============================
# Paths
# ===============================
DATA_DIR = r"D:\STT Summerizer project\dataset\ami_corpus_test"
GT_DIR = r"D:\STT Summerizer project\dataset\transcripts"
VOSK_OUT_DIR = r"D:\STT Summerizer project\dataset\stt_outputs\vosk"
WHISPER_OUT_DIR = r"D:\STT Summerizer project\dataset\stt_outputs\whisper"
VOSK_MODEL_PATH = r"D:\STT Summerizer project\model"
REPORT_DIR = r"D:\STT Summerizer project\dataset\reports"

os.makedirs(VOSK_OUT_DIR, exist_ok=True)
os.makedirs(WHISPER_OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

runtime_report_path = os.path.join(REPORT_DIR, "runtime_report.txt")
# clear old runtimes if file already exists
open(runtime_report_path, "w", encoding="utf-8").close()

if not os.path.exists(VOSK_MODEL_PATH):
    print(f"Please download a Vosk model from https://alphacephei.com/vosk/models and extract to {VOSK_MODEL_PATH}")
    exit(1)

# ===============================
# Load Models
# ===============================
whisper_model = whisper.load_model("base")
vosk_model = Model(VOSK_MODEL_PATH)
punct_model = PunctuationModel()  # For restoring punctuation in Vosk output

# ===============================
# Helper: clean text into readable form
# ===============================
def punctuate_text(text: str) -> str:
    """Restore punctuation and add line breaks after sentences."""
    if not text.strip():
        return text
    punctuated = punct_model.restore_punctuation(text)
    formatted = re.sub(r'([.?!])\s+', r'\1\n', punctuated)
    return formatted.strip()

# ===============================
# Main Loop
# ===============================
for wav_file in glob(os.path.join(DATA_DIR, "*.wav")):
    fname = os.path.splitext(os.path.basename(wav_file))[0]
    print(f"\nProcessing file: {wav_file}")

    vosk_runtime, whisper_runtime = None, None

    # ---- VOSK ----
    transcript = ""
    try:
        vosk_start = time.time()

        wf = wave.open(wav_file, "rb")
        print(f"Channels: {wf.getnchannels()}, Sample width: {wf.getsampwidth()}, Frame rate: {wf.getframerate()}")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print(f"⚠️ Audio format issue with {wav_file}. Vosk works best with mono, 16-bit WAV.")
        
        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                transcript += res.get("text", "") + " "
        res = json.loads(rec.FinalResult())
        transcript += res.get("text", "")
        transcript = transcript.strip()

        vosk_end = time.time()
        vosk_runtime = vosk_end - vosk_start

        # Add punctuation + formatting
        transcript = punctuate_text(transcript)

        print(f"Vosk transcript (first 100 chars): '{transcript[:100]}'")
        out_path = os.path.join(VOSK_OUT_DIR, fname + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcript)

    except Exception as e:
        print("Vosk error:", wav_file, e)
        transcript = ""

    # ---- WHISPER ----
    try:
        whisper_start = time.time()

        result = whisper_model.transcribe(wav_file)
        cleaned_text = re.sub(r'([.?!])\s+', r'\1\n', result["text"].strip())

        whisper_end = time.time()
        whisper_runtime = whisper_end - whisper_start

        print(f"Whisper transcript (first 100 chars): '{cleaned_text[:100]}'")
        out_path = os.path.join(WHISPER_OUT_DIR, fname + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
    except Exception as e:
        print("Whisper error:", wav_file, e)

    # ---- WER ----
    gt_path = os.path.join(GT_DIR, fname + ".txt")
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as f:
            gt = f.read().strip()
        with open(os.path.join(VOSK_OUT_DIR, fname + ".txt"), encoding="utf-8") as f:
            v_res = f.read().strip()
        with open(os.path.join(WHISPER_OUT_DIR, fname + ".txt"), encoding="utf-8") as f:
            w_res = f.read().strip()
        vosk_wer_val = wer(gt, v_res) if v_res else 1.0
        whisper_wer_val = wer(gt, w_res) if w_res else 1.0
        print(f"{fname} - VOSK WER: {vosk_wer_val:.3f}, Whisper WER: {whisper_wer_val:.3f}")
    else:
        print(f"GT file missing: {gt_path}")

    # ---- Save runtimes ----
    with open(runtime_report_path, "a", encoding="utf-8") as f:
        f.write(f"{fname} | Whisper: {whisper_runtime:.2f} sec | Vosk: {vosk_runtime:.2f} sec\n")

print(f"\n✅ STT processing + WER + runtime logging done. Runtimes saved to {runtime_report_path}")
