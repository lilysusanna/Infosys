#!/usr/bin/env python3
"""
evaluate_vosk.py
Transcribe a WAV file with Vosk and compute WER against a ground-truth text file.
Automatically saves evaluation results to results/vosk_evaluation_results.csv
"""
import argparse
import os
import wave
import json
import string
import csv
from vosk import Model, KaldiRecognizer
from jiwer import wer

# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def transcribe_wav(model_path, wav_path):
    wf = wave.open(wav_path, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("⚠️ WARNING: WAV should be 16kHz, mono, 16-bit. Consider resampling with ffmpeg:")
        print("   ffmpeg -i input.wav -ac 1 -ar 16000 -sample_fmt s16 output_16k_mono.wav")

    model = Model(model_path)
    rec = KaldiRecognizer(model, wf.getframerate())
    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            results.append(r.get("text", ""))
    r = json.loads(rec.FinalResult())
    results.append(r.get("text", ""))

    return " ".join(results).strip()

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to vosk model directory")
    parser.add_argument("--wav", required=True, help="Path to WAV file (16kHz mono)")
    parser.add_argument("--gt", required=True, help="Path to ground truth text file")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print("❌ Model path not found:", args.model)
        return
    if not os.path.isfile(args.wav):
        print("❌ WAV not found:", args.wav)
        return
    if not os.path.isfile(args.gt):
        print("❌ Ground truth file not found:", args.gt)
        return

    # --- Read GT safely ---
    try:
        with open(args.gt, "r", encoding="utf-8-sig") as f:
            gt = f.read().strip()
    except Exception as e:
        print("❌ Error reading ground truth:", e)
        return

    # --- Transcribe ---
    hyp = transcribe_wav(args.model, args.wav)

    # --- Normalize ---
    gt_norm = normalize_text(gt)
    hyp_norm = normalize_text(hyp)

    # --- Compute WER ---
    score = wer(gt_norm, hyp_norm)

    print("\n--- Evaluation Results ---")
    print("GROUND TRUTH (raw):\n", gt, "\n")
    print("HYPOTHESIS (raw):\n", hyp, "\n")
    print("GROUND TRUTH (normalized):\n", gt_norm, "\n")
    print("HYPOTHESIS (normalized):\n", hyp_norm, "\n")
    print(f"✅ WER = {score*100:.2f}%\n")

    # --- Save to CSV ---
    os.makedirs("results", exist_ok=True)
    csv_file = "results/vosk_evaluation_results.csv"

    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_file", "ground_truth", "hypothesis", "WER"])

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([args.wav, gt, hyp, f"{score*100:.2f}%"])

    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
