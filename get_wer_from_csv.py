import pandas as pd

def get_wer_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        
        # Manually process the 'wer' column to get only valid floats.
        # This handles rows with non-numeric values like "ERROR".
        valid_wers = []
        for wer_value in df['wer']:
            try:
                # Remove the '%' and convert to a float
                wer_float = float(str(wer_value).strip('%'))
                valid_wers.append(wer_float)
            except (ValueError, AttributeError):
                # Ignore values that are not valid numbers, like "ERROR"
                pass
        
        # Check if any valid values were found
        if not valid_wers:
            print(f"❌ Error: No valid WER values were found in the file '{csv_file}'.")
            return

        # Calculate the average from the valid values
        average_wer = sum(valid_wers) / len(valid_wers)
        print(f"\n✅ The overall Whisper WER from the saved file is: {average_wer:.2f}%\n")

    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_file}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    get_wer_from_csv("whisper_evaluation_results.csv")