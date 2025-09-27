import os
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import argparse

def transcript_to_annotation(file_path):
    """
    Converts a transcript txt file into an Annotation object.
    Expects each line in format:
    [SPEAKER_XX]: transcript text
    """
    annotation = Annotation()
    start = 0.0
    with open(file_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                speaker, text = line.split("]:", 1)
                speaker = speaker.replace("[", "")
                # Estimate duration (optional: 0.5 sec per word)
                duration = max(2.0, len(text.split()) * 0.5)
                end = start + duration
                annotation[Segment(start, end)] = speaker
                start = end
            except Exception as e:
                continue
    return annotation

def calculate_der(reference_file, hypothesis_file):
    reference = transcript_to_annotation(reference_file)
    hypothesis = transcript_to_annotation(hypothesis_file)

    der_metric = DiarizationErrorRate()
    der_score = der_metric(reference, hypothesis)
    return der_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        print(f"❌ Reference file not found: {args.reference}")
    elif not os.path.isfile(args.hypothesis):
        print(f"❌ Hypothesis file not found: {args.hypothesis}")
    else:
        der = calculate_der(args.reference, args.hypothesis)
        print(f"✅ Diarization Error Rate (DER): {der * 100:.2f}%")
        print(f"Target is < 20%")
