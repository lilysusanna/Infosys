#!/usr/bin/env python3
"""
evaluate_whisper.py
Transcribe a WAV file with Whisper and compute WER against a ground-truth text file.
Automatically saves evaluation results to results/whisper_evaluation_results.csv

Usage:
  python evaluate_whisper.py --model tiny.en --wav data/test_16k_mono.wav --gt data/test_16k_mono.txt
"""
import argparse
import os
import jiwer
import whisper
import string
import csv

# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base.en", help="Whisper model size (e.g., tiny.en, base.en, small.en)")
    parser.add_argument("--wav", required=True, help="Path to WAV file")
    parser.add_argument("--gt", required=True, help="Path to ground truth text file")
    args = parser.parse_args()

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

    # --- Load Whisper Model ---
    print(f"Loading Whisper model '{args.model}'...")
    try:
        model = whisper.load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # --- Transcribe ---
    print(f"Transcribing '{args.wav}'...")
    hyp = model.transcribe(args.wav, language='en')["text"]

    # --- Normalize and Compute WER ---
    gt_norm = normalize_text(gt)
    hyp_norm = normalize_text(hyp)
    score = jiwer.wer(gt_norm, hyp_norm)

    print("\n--- Evaluation Results ---")
    print("GROUND TRUTH (raw):\n", gt, "\n")
    print("HYPOTHESIS (raw):\n", hyp, "\n")
    print("GROUND TRUTH (normalized):\n", gt_norm, "\n")
    print("HYPOTHESIS (normalized):\n", hyp_norm, "\n")
    print(f"✅ WER = {score*100:.2f}%\n")

    # --- Save to CSV ---
    os.makedirs("results", exist_ok=True)
    csv_file = "results/whisper_evaluation_results.csv"

    # Write header if file doesn't exist
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["wav_file", "ground_truth", "hypothesis", "WER"])

    # Append current result
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([args.wav, gt, hyp, f"{score*100:.2f}%"])

    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
