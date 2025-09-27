from pyannote.audio import Pipeline
import os

def diarize_audio(audio_file):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(audio_file)

    stt_file = f"results/transcripts/{os.path.basename(audio_file).replace('.wav','')}_stt.txt"
    try:
        with open(stt_file) as f:
            stt_text = f.read().splitlines()
    except:
        stt_text = [""]

    diarized_transcript = []
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        text = stt_text[i] if i < len(stt_text) else ""
        diarized_transcript.append(f"[Speaker {speaker}]: {text}")

    out_file = f"results/transcripts/{os.path.basename(audio_file).replace('.wav','')}_diarized.txt"
    with open(out_file, "w") as f:
        for line in diarized_transcript:
            f.write(line + "\n")
    
    print(f"Diarized transcript saved at: {out_file}")
    return diarized_transcript
