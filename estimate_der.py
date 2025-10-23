import os

# Folder with diarized .txt files
DIARIZED_FOLDER = r"D:\yesu\STT Summerizer project\Diarization reports"
OUTPUT_FILE = os.path.join(DIARIZED_FOLDER, "der_results.txt")

# Function to load a diarized transcript
def load_transcript(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    transcript = []
    for line in lines:
        line = line.strip()
        if line:
            # Format: [SPEAKER_X] text
            if line.startswith("[") and "]" in line:
                speaker = line[1:line.index("]")]
                text = line[line.index("]")+1:].strip()
                transcript.append((speaker, text))
    return transcript

# Simple DER estimate between two transcripts
def estimate_der(trans1, trans2):
    # Just a naive estimate: proportion of words with different speakers
    words1 = []
    for speaker, text in trans1:
        words1.extend([(word, speaker) for word in text.split()])
    
    words2 = []
    for speaker, text in trans2:
        words2.extend([(word, speaker) for word in text.split()])
    
    min_len = min(len(words1), len(words2))
    if min_len == 0:
        return 0.0
    
    errors = sum(1 for i in range(min_len) if words1[i][1] != words2[i][1])
    der = errors / min_len
    return der

# Get all .txt diarized files
files = [f for f in os.listdir(DIARIZED_FOLDER) if f.endswith(".txt")]

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    out.write("DER Results\n\n")
    overall_ders = []

    for i, f1 in enumerate(files):
        trans1 = load_transcript(os.path.join(DIARIZED_FOLDER, f1))
        file_ders = []
        print(f"Processing file {i+1}/{len(files)}: {f1}")
        
        for j, f2 in enumerate(files):
            if f1 == f2:
                continue
            trans2 = load_transcript(os.path.join(DIARIZED_FOLDER, f2))
            der = estimate_der(trans1, trans2)
            file_ders.append(der)
            out.write(f"DER between {f1} and {f2}: {der:.2f}\n")
        
        if file_ders:
            avg_der = sum(file_ders)/len(file_ders)
            out.write(f"Average DER for {f1}: {avg_der:.2f}\n\n")
            overall_ders.append(avg_der)
    
    if overall_ders:
        total_avg = sum(overall_ders)/len(overall_ders)
        out.write(f"Overall average DER: {total_avg:.2f}\n")
    
print(f"\nDone! Results saved to {OUTPUT_FILE}")
