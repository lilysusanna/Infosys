import pandas as pd
import jiwer

def calculate_average_wer(csv_file):
    df = pd.read_csv(csv_file)

    # The 'wer' column in your CSV is a string like "2.35%". 
    # We need to convert it to a float.
    df['wer_float'] = df['wer'].str.strip('%').astype(float) / 100

    # Now, calculate the average of that new column
    average_wer = df['wer_float'].mean()

    print(f"\n✅ The average WER for the entire dataset is: {average_wer * 100:.2f}%")

if __name__ == "__main__":
    calculate_average_wer("evaluation_results.csv")