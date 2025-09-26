import pandas as pd
import jiwer
import os
import argparse

def compute_and_save_wer(hyp_file, ref_file, csv_file="evaluation_results.csv"):
    # Read files
    with open(hyp_file, "r", encoding="utf-8") as f:
        hyp_text = f.read().replace("\n", " ").strip()
    with open(ref_file, "r", encoding="utf-8") as f:
        ref_text = f.read().replace("\n", " ").strip()
    
    # Compute WER
    wer_value = jiwer.wer(ref_text, hyp_text)
    wer_percent = wer_value * 100

    # Save to CSV
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["hyp", "ref", "wer"])

    df = pd.concat([df, pd.DataFrame([{
        "hyp": hyp_text,
        "ref": ref_text,
        "wer": f"{wer_percent:.2f}%"
    }])], ignore_index=True)

    df.to_csv(csv_file, index=False)
    print(f"✅ WER saved to {csv_file} -> {wer_percent:.2f}%")

    return wer_percent

def average_wer(csv_file="evaluation_results.csv"):
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found.")
        return
    df = pd.read_csv(csv_file)
    df['wer_float'] = df['wer'].str.strip('%').astype(float) / 100
    avg = df['wer_float'].mean() * 100
    print(f"\n📊 Average WER for all transcripts: {avg:.2f}%\n")
    return avg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp", required=True, help="Hypothesis transcript (.txt)")
    parser.add_argument("--ref", required=True, help="Reference transcript (.txt)")
    parser.add_argument("--csv", default="evaluation_results.csv", help="CSV to save results")
    args = parser.parse_args()

    compute_and_save_wer(args.hyp, args.ref, args.csv)
    average_wer(args.csv)
