import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import re
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer

def clean_text(text):
    cleaned = re.sub(r"\[\w+\]:", "", text)
    cleaned = " ".join(cleaned.split())
    return cleaned

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return "\n".join(sentences)

def summarize_text(summarizer, text):
    cleaned_text = clean_text(text)
    max_chars = 1500
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars]
    outputs = summarizer(cleaned_text, max_length=256, min_length=64, do_sample=False)
    return outputs[0]['summary_text']

def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

print("Loading summarization pipeline (BART)...")
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")

reference_summary = "We discussed next quarter goals and planned a 20% sales increase. Budget needs review and a report will be prepared next week."

summary_folder = "Diarization reports"
files_found = 0
for filename in os.listdir(summary_folder):
    if filename.startswith("diarized_") and filename.endswith(".wav.txt"):
        files_found += 1
        diarized_path = os.path.join(summary_folder, filename)
        with open(diarized_path, "r", encoding="utf-8") as f:
            diarized_transcript = f.read()
        print(f"\nProcessing file: {diarized_path}")
        summary = summarize_text(summarizer, diarized_transcript)
        formatted_summary = split_into_sentences(summary)
        print("Summary:\n", formatted_summary)
        summary_file = os.path.join(summary_folder, filename.replace("diarized_", "summary_"))
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(formatted_summary)
        print(f"Saved summary to {summary_file}")
        # Evaluate ROUGE
        scores = evaluate_rouge(reference_summary, summary)
        print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.3f}, ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}")
        eval_file = os.path.join(summary_folder, filename.replace("diarized_", "eval_"))
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.3f}\n")
            f.write(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}\n")
        print(f"Saved evaluation scores to {eval_file}")

if files_found == 0:
    print(f"No 'diarized_*.wav.txt' files found in folder '{summary_folder}'. Please check the folder path and file names.")

print("Summarization complete.")
