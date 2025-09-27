from transformers import pipeline
import os

def summarize_diarized_transcript(file_path):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    with open(file_path, "r") as f:
        text = f.read()

    prompt = "Summarize this meeting transcript while preserving speaker structure:\n\n" + text

    summary = summarizer(prompt, max_length=200, min_length=50, do_sample=False)

    out_file = file_path.replace("_diarized.txt", "_summary.txt")
    with open(out_file, "w") as f:
        f.write(summary[0]['summary_text'])
    
    print(f"Summary saved at: {out_file}")
    return summary[0]['summary_text']