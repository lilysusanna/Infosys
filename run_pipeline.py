import argparse, os
from diarize_and_transcribe import align_and_save
from summarizer import summarize_transcript
from utils import read_text, write_text
from config import SUMMARIES_DIR

def run(audio, template="generic_meeting", summarize=True):
    transcript_path = align_and_save(audio)
    if summarize:
        txt = read_text(transcript_path)
        summary = summarize_transcript(txt, template=template)
        out_sum = os.path.join(SUMMARIES_DIR, os.path.basename(audio) + ".summary.txt")
        write_text(out_sum, summary)
        print("Summary saved:", out_sum)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--template", default="generic_meeting")
    parser.add_argument("--no-summary", action="store_true")
    args = parser.parse_args()
    run(args.audio, template=args.template, summarize=not args.no_summary)
