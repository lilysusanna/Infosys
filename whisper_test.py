import whisper
import os

# --- Configuration ---
# Choose a model size. 'tiny' is fastest, 'base' is a good balance.
# For full accuracy on diverse accents, you might need 'small' or 'medium'.
# The .en models are English-only and are faster.
model_size = "base.en"

# Path to your test audio file
audio_file = os.path.join(os.path.dirname(__file__), "..", "data", "test_16k_mono.wav")

if not os.path.exists(audio_file):
    print(f"❌ Audio file not found: {audio_file}")
    print("Please make sure you have a test file named 'test_16k_mono.wav' in your data/ folder.")
else:
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Transcribing audio file '{audio_file}'...")
    result = model.transcribe(audio_file)

    print("\n--- Transcription Result ---")
    print(f"Text: {result['text']}")
    print(f"Language: {result['language']}")