import os
from pyannote.audio import Pipeline
from pyannote.core import Annotation
import whisper
import torch
import soundfile as sf
import argparse
import numpy as np

# --- Configuration ---
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HUGGING_FACE_TOKEN:
    print("❌ Error: Hugging Face access token not found.")
    print("Please set the HUGGING_FACE_HUB_TOKEN environment variable.")
    exit()

# Whisper model to use for transcription
WHISPER_MODEL_SIZE = "base.en"

# --- Main Logic ---
def diarize_and_transcribe(audio_path, whisper_model_size):
    # 1. Load Pyannote Diarization Pipeline
    print("Loading Pyannote Diarization model...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )

    # 2. Run Diarization on the audio file
    print(f"Running diarization on {audio_path}...")
    diarization_result = diarization_pipeline(audio_path)

    # 3. Load Whisper Transcription Model
    print(f"Loading Whisper model '{whisper_model_size}'...")
    whisper_model = whisper.load_model(whisper_model_size)

    # 4. Process each speaker's segment
    transcriptions = []
    
    # Read the full audio file into memory
    full_audio, _ = sf.read(audio_path)

    print("\nStarting transcription and diarization...")
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        start_time, end_time = turn.start, turn.end
        start_sample = int(start_time * 16000)
        end_sample = int(end_time * 16000)
        
        # Extract the audio segment for the current speaker
        segment_audio = full_audio[start_sample:end_sample]

        # Transcribe the audio segment with Whisper, ensuring the correct data type
        segment_result = whisper_model.transcribe(segment_audio.astype(np.float32))
        segment_text = segment_result["text"].strip()
        
        # Format the output as "[Speaker 1]: Let's discuss..."
        transcriptions.append(f"[{speaker}]: {segment_text}")
    
    # --- Save the outputs ---
    # Save the diarization result to an RTTM file for evaluation
    output_rttm_path = "output.rttm"
    with open(output_rttm_path, "w") as f:
        diarization_result.write_rttm(f)
        
    print(f"✅ Diarization output saved to {output_rttm_path}")
    
    # Save the full transcript to a separate text file
    output_transcript_path = "diarized_transcript.txt"
    full_transcript = "\n".join(transcriptions)
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write(full_transcript)
    print(f"✅ Full diarized transcript saved to {output_transcript_path}")

    return full_transcript

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize and transcribe an audio file.")
    parser.add_argument("--audio", required=True, help="Path to the audio file.")
    parser.add_argument("--model", default=WHISPER_MODEL_SIZE, help="Whisper model size.")
    args = parser.parse_args()

    # The script expects a 16kHz mono WAV file
    if not os.path.isfile(args.audio):
        print(f"❌ Error: Audio file not found at {args.audio}")
        exit()
    try:
        audio_info = sf.info(args.audio)
        if audio_info.samplerate != 16000:
            print("❌ Error: Audio file must be 16kHz.")
            exit()
        if audio_info.channels != 1:
            print("❌ Error: Audio file must be mono.")
            exit()
    except Exception as e:
        print(f"❌ Error reading audio file info: {e}")
        exit()

    final_transcript = diarize_and_transcribe(args.audio, args.model)
    
    print("\n--- Final Diarized Transcript ---")
    print(final_transcript)