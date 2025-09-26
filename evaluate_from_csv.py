import pandas as pd
import argparse
import whisper
import jiwer
import os

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate a model on a dataset from a CSV.")
    parser.add_argument("--model", default="base.en", help="Whisper model size.")
    parser.add_argument("--input_csv", default="dataset.csv", help="Path to the dataset CSV.")
    args = parser.parse_args()

    # Load the Whisper model
    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model)

    # Load the dataset from the CSV
    df = pd.read_csv(args.input_csv)

    # Convert FLAC paths to WAV paths for evaluation
    df['audio_path'] = df['audio_path'].str.replace('.flac', '.wav', regex=False)

    # Add a column for the Whisper hypothesis
    df['hypothesis'] = ''

    print("Starting transcription...")
    for index, row in df.iterrows():
        print(f"Transcribing {index + 1}/{len(df)}: {row['audio_path']}")
        try:
            # Check if the WAV file exists before transcribing
            if not os.path.isfile(row['audio_path']):
                print(f"❌ WAV file not found, skipping: {row['audio_path']}")
                df.at[index, 'hypothesis'] = "FILE_NOT_FOUND_ERROR"
                continue

            result = model.transcribe(row['audio_path'], language='en')
            df.at[index, 'hypothesis'] = result['text']
        except Exception as e:
            print(f"❌ Error during transcription: {e}, skipping.")
            df.at[index, 'hypothesis'] = "TRANSCRIPTION_ERROR"

    # Compute WER for each row
    df['wer'] = df.apply(lambda row: jiwer.wer(
        row['normalized_gt'], row['hypothesis']
    ), axis=1)

    # Save the results
    output_filename = f"whisper_evaluation_results.csv"
    df.to_csv(output_filename, index=False)

    # Calculate and print overall WER
    overall_wer = df['wer'].mean()
    print(f"\n✅ Batch evaluation complete. Overall WER: {overall_wer * 100:.2f}%")
    print(f"Detailed results saved to {output_filename}")

if __name__ == "__main__":
    main()