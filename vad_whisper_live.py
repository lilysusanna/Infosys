import os
import whisper
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import time

# --- Configuration ---
# Your preferred Whisper model. 'tiny.en' or 'base.en' is good for real-time.
WHISPER_MODEL_SIZE = "base.en" 
# How many blocks of silence to wait before finalizing an utterance.
SILENCE_BLOCKS = 3  # e.g., 3 * 500ms = 1.5 seconds of silence
# The audio block size (in samples).
BLOCK_SIZE = 512# 8000 samples @ 16kHz = 0.5 seconds of audio
SAMPLE_RATE = 16000

# --- VAD Setup ---
# Use a more robust way to load the VAD model.
try:
    print("Attempting to load Silero VAD model via torch.hub...")
    VAD_MODEL, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                       model='silero_vad',
                                       force_reload=False)
except Exception as e:
    print(f"❌ Error loading Silero VAD model via torch.hub: {e}")
    print("Trying alternate import method...")
    try:
        from silero_vad import load_silero_vad
        VAD_MODEL, utils = load_silero_vad()
    except Exception as e:
        print(f"❌ Critical Error: Could not load Silero VAD model. Ensure `pip install silero-vad` succeeded.")
        print(e)
        exit()

VAD_MODEL.eval()

# Helper function to check if a single audio chunk is speech or silence
def is_speech(chunk, model, sample_rate):
    """
    Check if an audio chunk contains speech.
    Returns True for speech, False for silence.
    """
    with torch.no_grad():
        wav_tensor = torch.from_numpy(chunk).float()
        speech_prob = model(wav_tensor, sample_rate).item()
        return speech_prob > 0.5

# --- Main Logic ---
def main():
    try:
        # Load Whisper model
        print(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
        whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        
        print("\nListening for speech... (Press Ctrl+C to exit)")

        audio_buffer = np.array([], dtype=np.int16)
        silent_blocks_count = 0
        
        def audio_callback(indata, frames, time, status):
            nonlocal audio_buffer, silent_blocks_count
            
            # Use the VAD model to check the current chunk
            is_speaking = is_speech(indata.flatten().astype(np.float32) / 32768.0, VAD_MODEL, SAMPLE_RATE)

            if is_speaking:
                # If speech is detected, reset the silence counter and append to buffer
                silent_blocks_count = 0
                audio_buffer = np.append(audio_buffer, indata.flatten())
            else:
                # If silence, increment the counter
                silent_blocks_count += 1
                
            # If a long period of silence is detected, transcribe the accumulated audio
            if silent_blocks_count >= SILENCE_BLOCKS and len(audio_buffer) > 0:
                print("\nTranscribing utterance...")
                try:
                    # Convert buffer to a format Whisper can use
                    buffer_tensor = torch.from_numpy(audio_buffer).float() / 32768.0
                    
                    # Transcribe the audio buffer
                    result = whisper_model.transcribe(buffer_tensor, language='en')
                    transcribed_text = result["text"].strip()
                    
                    if transcribed_text:
                        print("✅ Final Transcription:", transcribed_text)
                        # Save to a transcript file (optional, similar to realtime_stt.py)
                        # os.makedirs(os.path.join(os.path.dirname(__file__), "..", "transcripts"), exist_ok=True)
                        # with open("transcripts/realtime_log.txt", "a") as f:
                        #     f.write(transcribed_text + "\n")

                except Exception as e:
                    print(f"❌ Transcription error: {e}")
                
                # Reset the buffer and silence counter
                audio_buffer = np.array([], dtype=np.int16)
                silent_blocks_count = 0
                print("\nListening for next utterance...")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16',
                            blocksize=BLOCK_SIZE, callback=audio_callback):
            print("Press Ctrl+C to stop.")
            while True:
                time.sleep(1) # Keep main thread alive
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()