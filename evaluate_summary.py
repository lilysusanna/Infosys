import argparse
import os
import csv
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def evaluate_rouge(system_summary, reference_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_summary, system_summary)
    return scores

def evaluate_bleu(system_summary, reference_summary):
    reference_tokens = [reference_summary.split()]
    system_tokens = system_summary.split()
    smoothing = SmoothingFunction().method1
    score = sentence_bleu(reference_tokens, system_tokens, smoothing_function=smoothing)
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, help="Path to system-generated summary")
    parser.add_argument("--reference", required=True, help="Path to reference summary")
    args = parser.parse_args()

    # Read summaries
    with open(args.system, "r", encoding="utf-8") as f:
        system_summary = f.read().strip()
    with open(args.reference, "r", encoding="utf-8") as f:
        reference_summary = f.read().strip()

    # Evaluate
    rouge_scores = evaluate_rouge(system_summary, reference_summary)
    bleu_score = evaluate_bleu(system_summary, reference_summary)

    rouge1_f1 = round(rouge_scores['rouge1'].fmeasure, 4)
    rougeL_f1 = round(rouge_scores['rougeL'].fmeasure, 4)
    bleu = round(bleu_score, 4)

    # Print results
    print("✅ ROUGE-1 F1 Score:", rouge1_f1)
    print("✅ ROUGE-L F1 Score:", rougeL_f1)
    print("✅ BLEU Score:", bleu)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    csv_file = os.path.join("results", "evaluate_RougeResults.csv")

    # If CSV exists, append; otherwise, create and write header
    write_header = not os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["System Summary", "Reference Summary", "ROUGE-1 F1", "ROUGE-L F1", "BLEU"])
        writer.writerow([os.path.basename(args.system), os.path.basename(args.reference), rouge1_f1, rougeL_f1, bleu])

    print(f"✅ Results saved to {csv_file}")
