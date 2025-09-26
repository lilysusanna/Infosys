import os
import csv
import argparse
import string

def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and trim extra spaces."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def create_dataset_csv(input_dir, output_csv):
    """Parses LibriSpeech dataset and saves audio paths and transcripts to a CSV."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['audio_path', 'ground_truth', 'normalized_gt'])

        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.trans.txt'):
                    trans_path = os.path.join(root, file)
                    print(f"Parsing transcript file: {trans_path}")
                    with open(trans_path, 'r', encoding='utf-8') as trans_f:
                        for line in trans_f:
                            parts = line.strip().split(' ', 1)
                            if len(parts) == 2:
                                file_id, text = parts
                                # The original audio file is a .flac, but we'll assume a .wav
                                # is created in the same location.
                                audio_file = f"{file_id}.flac"
                                audio_path = os.path.join(root, audio_file)
                                
                                normalized_text = normalize_text(text)
                                writer.writerow([audio_path, text, normalized_text])

    print(f"\n✅ Dataset CSV created successfully at {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a CSV from LibriSpeech transcripts.")
    parser.add_argument("--input", required=True, help="Path to the root of the LibriSpeech dataset.")
    parser.add_argument("--output", default="dataset.csv", help="Output CSV file path.")
    args = parser.parse_args()
    create_dataset_csv(args.input, args.output)