#!/usr/bin/env python3
"""
batch_evaluate_vosk.py
Batch-transcribe all WAV files in a LibriSpeech-like directory structure.
This version correctly handles the single transcript file for each folder.


Usage:
  python scripts/batch_evaluate_vosk.py --model models/vosk-model-en-us-0.22 --input data/LibriSpeech/test-clean
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

def transcribe_wav(model, wav_path):
    """Transcribe a single WAV file using Vosk."""
    try:
        wf = wave.open(wav_path, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            print(f"  ⚠️ Skipping {os.path.basename(wav_path)}: not 16kHz mono 16-bit.")
            return None
    except wave.Error as e:
        print(f"  ❌ Error opening {os.path.basename(wav_path)}: {e}")
        return None

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

def get_transcripts_from_file(transcript_path):
    """Parses a LibriSpeech-style transcript file into a dictionary."""
    transcripts = {}
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                file_id, text = parts
                transcripts[file_id] = text
    return transcripts

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Batch evaluate Vosk on a dataset folder.")
    parser.add_argument("--model", required=True, help="Path to Vosk model directory.")
    parser.add_argument("--input", required=True, help="Input directory containing audio and ground truth files.")
    parser.add_argument("--output", default="evaluation_results.csv", help="Output CSV file for results.")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print("❌ Model path not found:", args.model)
        return
    if not os.path.isdir(args.input):
        print("❌ Input directory not found:", args.input)
        return

    # Initialize Vosk model once
    print("Loading Vosk model...")
    model = Model(args.model)

    results_data = []
    
    # Walk through the input directory
    for root, dirs, files in os.walk(args.input):
        # Look for a transcript file in the current directory
        trans_file_path = None
        for file in files:
            if file.endswith('.txt') and not file.startswith('.'):
                trans_file_path = os.path.join(root, file)
                break
        
        # If a transcript file is found, parse it
        transcripts = {}
        if trans_file_path:
            transcripts = get_transcripts_from_file(trans_file_path)

        # Process the WAV files in the current directory
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                
                # Use the file's ID to find its transcript
                gt = transcripts.get(base_name)
                if not gt:
                    print(f"  ⚠️ Skipping {file}: no matching transcript found in {trans_file_path or 'any transcript file'}.")
                    continue
                
                print(f"Processing {file}...")
                
                # Transcribe
                hyp = transcribe_wav(model, wav_path)
                if hyp is None:
                    continue

                # Normalize and compute WER
                gt_norm = normalize_text(gt)
                hyp_norm = normalize_text(hyp)
                score = wer(gt_norm, hyp_norm)
                
                results_data.append({
                    "filename": file,
                    "wer": f"{score*100:.2f}%",
                    "ground_truth": gt,
                    "hypothesis": hyp
                })

    # Write results to CSV
    if results_data:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "wer", "ground_truth", "hypothesis"])
            writer.writeheader()
            writer.writerows(results_data)
        print(f"\n✅ Batch evaluation complete. Results saved to {args.output}")
    else:
        print("\nNo WAV files processed. Check your --input path and file formats.")

if __name__ == "__main__":
    main()