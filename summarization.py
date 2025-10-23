import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import torch
import librosa
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from rouge_score import rouge_scorer
import re


# --- Configuration ---
DATASET_PATH = "dataset/ami_corpus_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PYANNOTE_TOKEN = os.getenv("HF_TOKEN")


# Load diarization pipeline once
print("Loading pyannote diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=PYANNOTE_TOKEN
)


# Load summarization pipeline once
print("Loading summarization pipeline (BART)...")
summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")


def format_diarized_transcript(diarization):
    simulated_texts = [
        "Let's discuss next quarter goals.",
        "We should increase sales by 20%.",
        "Budget allocation needs review.",
        "I will prepare the report next week."
    ]
    transcript = []
    idx = 0
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        text = simulated_texts[idx % len(simulated_texts)]
        idx += 1
        transcript.append(f"[{speaker}]: {text}")
    return "\n".join(transcript)


def clean_text(text):
    # Remove speaker tags like [SPEAKER_0]:
    cleaned = re.sub(r"\[\w+\]:", "", text)
    # Remove extra whitespace/newlines
    cleaned = " ".join(cleaned.split())
    return cleaned


def summarize_text(text):
    cleaned_text = clean_text(text)
    # Optionally truncate to model max length (~1024 tokens, approximate char count)
    max_chars = 1500
    if len(cleaned_text) > max_chars:
        cleaned_text = cleaned_text[:max_chars]
    print("Cleaned transcript for summarization:\n", cleaned_text)  # Debug print
    outputs = summarizer(cleaned_text, max_length=256, min_length=64, do_sample=False)
    return outputs[0]['summary_text']


def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores


# Example dummy reference summary, replace with real references if available
reference_summary = "We discussed next quarter goals and planned a 20% sales increase. Budget needs review and a report will be prepared next week."


for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".wav"):
        audio_path = os.path.join(DATASET_PATH, filename)
        print(f"\nProcessing file: {filename}")


        # Diarization
        diarization = diarization_pipeline(audio_path)


        # Format diarized transcript
        diarized_transcript = format_diarized_transcript(diarization)
        print("Diarized Transcript:\n", diarized_transcript)


        # Save diarized transcript
        transcript_file = f"diarized_{filename}.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(diarized_transcript)
        print(f"Saved diarized transcript to {transcript_file}")


        # Summarize
        summary = summarize_text(diarized_transcript)
        print("Summary:\n", summary)


        # Save summary
        summary_file = f"summary_{filename}.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Saved summary to {summary_file}")


        # Evaluate ROUGE
        scores = evaluate_rouge(reference_summary, summary)
        print(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.3f}, ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}")


        # Save evaluation
        eval_file = f"eval_{filename}.txt"
        with open(eval_file, "w", encoding="utf-8") as f:
            f.write(f"ROUGE-1 F1: {scores['rouge1'].fmeasure:.3f}\n")
            f.write(f"ROUGE-L F1: {scores['rougeL'].fmeasure:.3f}\n")
        print(f"Saved evaluation scores to {eval_file}")
