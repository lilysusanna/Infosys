import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration ---
# Choose a summarization model. 't5-small' is good for testing.
MODEL_NAME = "t5-small"

def main():
    parser = argparse.ArgumentParser(description="Generate a summary from a diarized transcript.")
    parser.add_argument("--transcript", required=True, help="Path to the diarized transcript file.")
    args = parser.parse_args()

    # Make sure the transcript file exists
    if not os.path.isfile(args.transcript):
        print(f"❌ Error: Transcript file not found at {args.transcript}")
        exit()

    # Load the tokenizer and model (downloads automatically on first run)
    print(f"Loading tokenizer and model for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # Read the full diarized transcript from the file
    with open(args.transcript, "r", encoding="utf-8") as f:
        diarized_transcript = f.read()

    # Pre-process the text for the T5 model
    # T5 models require a specific prefix for the summarization task
    processed_text = "summarize: " + diarized_transcript
    
    # Tokenize the input text
    inputs = tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary
    print("Generating summary...")
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=150,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )
    
    # Decode the output
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    print("\n--- Generated Summary ---")
    print(summary)
    
    # --- Save the summary to a file ---
    output_summary_path = "summary_output.txt"
    with open(output_summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n✅ Summary saved to {output_summary_path}")

if __name__ == "__main__":
    main()