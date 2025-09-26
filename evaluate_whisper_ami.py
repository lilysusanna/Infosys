import os
import whisper
import jiwer
import pandas as pd

# Set paths
audio_folder = r"C:\Users\A.VYSHNAVI\.cache\kagglehub\datasets\bnarayanareddy\ami-corpus-test\versions\1\content\ami_corpus_test"
transcript_folder = r"C:\Users\A.VYSHNAVI\Desktop\infosys\transcripts"

# Map audio files to transcript filenames
audio_files = [
    "EN2002b.wav",
    "EN2002c.wav"
]

# Load corresponding transcripts (without timestamps)
transcripts = {
    "EN2002b.wav": os.path.join(transcript_folder, "EN2002b.txt"),
    "EN2002c.wav": os.path.join(transcript_folder, "EN2002c.txt")
}

# Load Whisper model
print("Loading Whisper model 'small.en'...")
model = whisper.load_model("small.en")
print("✅ Model loaded successfully.")

wers = []
results = []

for file in audio_files:
    audio_path = os.path.join(audio_folder, file)
    
    if not os.path.exists(transcripts[file]):
        print(f"⚠️  Transcript missing for {file}, skipping.")
        continue

    with open(transcripts[file], "r", encoding="utf-8") as f:
        reference_text = f.read().strip()

    # Transcribe using Whisper
    result = model.transcribe(audio_path, language='en')
    predicted_text = result["text"].strip()

    # Compute WER
    wer = jiwer.wer(reference_text, predicted_text) * 100
    wers.append(wer)

    results.append({
        "audio_file": file,
        "reference_text": reference_text,
        "predicted_text": predicted_text,
        "wer": wer
    })

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv("whisper_ami_evaluation.csv", index=False)

if wers:
    print(f"📊 Average WER: {sum(wers)/len(wers):.2f}%")
else:
    print("❌ No audio-transcript pairs processed.")
