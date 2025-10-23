import os
from jiwer import wer, Compose, RemovePunctuation, ToLowerCase, Strip, RemoveMultipleSpaces

# ===============================
# Paths
# ===============================
VOSK_OUT_DIR = r"D:\STT Summerizer project\dataset\stt_outputs\vosk"
WHISPER_OUT_DIR = r"D:\STT Summerizer project\dataset\stt_outputs\whisper"
GT_DIR = r"D:\STT Summerizer project\dataset\transcripts"
REPORT_DIR = r"D:\STT Summerizer project\dataset\reports"
os.makedirs(REPORT_DIR, exist_ok=True)

runtime_file = os.path.join(REPORT_DIR, "runtime_report.txt")

# ===============================
# Load runtimes if available
# ===============================
runtimes = {}
if os.path.exists(runtime_file):
    with open(runtime_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                fname, times = line.split(":")
                parts = times.split("|")
                whisper_time = float(parts[0].split()[1])
                vosk_time = float(parts[1].split()[1])
                runtimes[fname.strip()] = (whisper_time, vosk_time)
            except:
                continue
else:
    print("⚠️ No runtime_report.txt found. Runtime info will be skipped.")

# ===============================
# WER Transformation
# ===============================
transformation = Compose([
    RemovePunctuation(),
    ToLowerCase(),
    Strip(),
    RemoveMultipleSpaces()
])

# ===============================
# Initialize summary variables
# ===============================
total_files = 0
sum_wer_whisper_gt = 0
sum_wer_vosk_gt = 0
sum_wer_vosk_whisper = 0

# ===============================
# Generate Comparison Reports
# ===============================
for file in os.listdir(WHISPER_OUT_DIR):
    if not file.endswith(".txt"):
        continue

    fname = os.path.splitext(file)[0]
    whisper_file = os.path.join(WHISPER_OUT_DIR, file)
    vosk_file = os.path.join(VOSK_OUT_DIR, file)
    gt_file = os.path.join(GT_DIR, file)

    if not os.path.exists(vosk_file):
        print(f"⚠️ Skipping {fname}, missing Vosk transcript.")
        continue
    if not os.path.exists(gt_file):
        print(f"⚠️ Skipping {fname}, missing GT transcript.")
        continue

    with open(whisper_file, "r", encoding="utf-8") as f:
        whisper_text = f.read().strip()

    with open(vosk_file, "r", encoding="utf-8") as f:
        vosk_text = f.read().strip()

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_text = f.read().strip()

    # Apply transformations manually
    gt_norm = transformation(gt_text)
    whisper_norm = transformation(whisper_text)
    vosk_norm = transformation(vosk_text)

    # Compute WERs
    wer_whisper_gt = wer(gt_norm, whisper_norm) * 100
    wer_vosk_gt = wer(gt_norm, vosk_norm) * 100
    wer_vosk_whisper = wer(whisper_norm, vosk_norm) * 100

    # Determine better STT based on WER vs GT
    if wer_whisper_gt < wer_vosk_gt:
        better_stt = "Whisper ✅ Better"
    elif wer_vosk_gt < wer_whisper_gt:
        better_stt = "Vosk ✅ Better"
    else:
        better_stt = "Tie"

    # Update summary
    total_files += 1
    sum_wer_whisper_gt += wer_whisper_gt
    sum_wer_vosk_gt += wer_vosk_gt
    sum_wer_vosk_whisper += wer_vosk_whisper

    # Build per-file report
    report_lines = [
        f"📊 Comparison Report for {fname}",
        f"Whisper words: {len(whisper_text.split())}",
        f"Vosk words   : {len(vosk_text.split())}",
        f"WER (Whisper vs GT)   = {wer_whisper_gt:.2f}%",
        f"WER (Vosk vs GT)      = {wer_vosk_gt:.2f}%",
        f"WER (Vosk vs Whisper) = {wer_vosk_whisper:.2f}%",
        f"Better STT (based on WER vs GT) = {better_stt}"
    ]

    if fname in runtimes:
        wt, vt = runtimes[fname]
        report_lines.append(f"Runtime: Whisper {wt:.2f}s | Vosk {vt:.2f}s")

    report_lines.extend([
        "\n🔹 Sample Outputs:",
        "\nWhisper (first 200 chars):\n" + whisper_text[:200] + " ...",
        "\nVosk (first 200 chars):\n" + vosk_text[:200] + " ..."
    ])

    # Save per-file report
    report_file = os.path.join(REPORT_DIR, fname + "_comparison.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"✅ Report saved: {report_file}")

# ===============================
# Generate Final Summary Report
# ===============================
if total_files > 0:
    avg_whisper_gt = sum_wer_whisper_gt / total_files
    avg_vosk_gt = sum_wer_vosk_gt / total_files
    avg_vosk_whisper = sum_wer_vosk_whisper / total_files

    # Determine overall better STT
    if avg_whisper_gt < avg_vosk_gt:
        overall_better = "Whisper ✅ Better on average"
    elif avg_vosk_gt < avg_whisper_gt:
        overall_better = "Vosk ✅ Better on average"
    else:
        overall_better = "Tie"

    summary_lines = [
        "📊 Final Summary Report",
        f"Total files processed : {total_files}",
        f"Average WER (Whisper vs GT)   = {avg_whisper_gt:.2f}%",
        f"Average WER (Vosk vs GT)      = {avg_vosk_gt:.2f}%",
        f"Average WER (Vosk vs Whisper) = {avg_vosk_whisper:.2f}%",
        f"Overall Better STT             = {overall_better}"
    ]

    summary_file = os.path.join(REPORT_DIR, "final_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\n✅ Final summary report saved: {summary_file}")
else:
    print("⚠️ No files processed. Check transcripts directories.")
