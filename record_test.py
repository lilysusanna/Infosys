import sounddevice as sd
import soundfile as sf

# Recording settings
samplerate = 16000      # 16 kHz sample rate
duration = 5            # seconds to record
outfile = "data/test_16k_mono.wav"

print(f"Recording for {duration} seconds... Speak now 🎤")
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()  # wait until recording is finished
sf.write(outfile, recording, samplerate, subtype='PCM_16')
print("✅ Saved recording to:", outfile)
