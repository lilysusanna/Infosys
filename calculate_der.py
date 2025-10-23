import os
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation
import argparse

def calculate_der(reference_path, hypothesis_path):
    """
    Calculates the Diarization Error Rate (DER) by comparing a reference
    RTTM file to a hypothesis RTTM file.
    """
    # Load the reference (ground truth) RTTM file
    reference = Annotation.from_file(reference_path)
    
    # Load the hypothesis (your system's output) RTTM file
    hypothesis = Annotation.from_file(hypothesis_path)
    
    # Initialize the DER metric
    der_metric = DiarizationErrorRate()
    
    # Compute the DER
    der_score = der_metric(reference, hypothesis)
    
    return der_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Diarization Error Rate (DER).")
    parser.add_argument("--reference", required=True, help="Path to the ground truth .rttm file.")
    parser.add_argument("--hypothesis", required=True, help="Path to your system's output .rttm file.")
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        print(f"❌ Error: Reference file not found at {args.reference}")
    elif not os.path.isfile(args.hypothesis):
        print(f"❌ Error: Hypothesis file not found at {args.hypothesis}")
    else:
        der = calculate_der(args.reference, args.hypothesis)
        print(f"\n--- Diarization Evaluation Results ---")
        print(f"✅ Diarization Error Rate (DER): {der * 100:.2f}%")
        print(f"Target is < 20%")