# convert_txt_to_rttm.py

txt_file = "results/realtime_gt1.txt"      # Your GT text file
audio_filename = "test1"                   # Base name of your audio file
output_rttm = "results/realtime_gt1.rttm"  # Output RTTM file

# Read the text
with open(txt_file, "r") as f:
    text = f.read().strip()

# Estimate duration: 0.5 seconds per word
words = text.split()
duration = max(2.0, len(words) * 0.5)  # minimum 2 sec

start_time = 0.0
end_time = start_time + duration

# Write RTTM
with open(output_rttm, "w") as f:
    f.write(f"SPEAKER {audio_filename} 1 {start_time:.2f} {duration:.2f} <NA> <NA> SPEAKER_00 <NA> <NA>\n")

print(f"✅ RTTM created: {output_rttm}")
