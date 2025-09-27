from milestone2.diarization_module import diarize_audio
from milestone2.summarization_module import summarize_diarized_transcript

audio_files = ["audio_wav/test1.wav", "audio_wav/test2.wav", "audio_wav/test3.wav"]

for audio_file in audio_files:
    print(f"\nProcessing {audio_file} ...")
    diarize_audio(audio_file)
    summarize_diarized_transcript(f"results/transcripts/{audio_file.split('/')[-1].replace('.wav','')}_diarized.txt")
