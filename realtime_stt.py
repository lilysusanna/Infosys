import argparse
import os
import queue
import sys
import json
import datetime
import sounddevice as sd
from vosk import Model, KaldiRecognizer

q = queue.Queue()

def audio_callback(indata, frames, time, status):
    # indata is bytes (RawInputStream) or numpy array depending on stream type;
    # we convert to bytes to feed KaldiRecognizer.
    if status:
        print("Status:", status, file=sys.stderr)
    q.put(bytes(indata))

def list_devices():
    print(sd.query_devices())

def main():
    parser = argparse.ArgumentParser(description="Realtime STT with Vosk")
    parser.add_argument("--model", required=True, help="Path to vosk model folder (e.g. ../models/vosk-model-small-en-us-0.15)")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sample rate (default 16000)")
    parser.add_argument("--device", type=int, default=None, help="Sounddevice input device index (use --list-devices to find indices)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--blocksize", type=int, default=8000, help="Block size in frames (default 8000 -> 0.5s at 16kHz)")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not os.path.isdir(args.model):
        print("Model path not found:", args.model)
        sys.exit(1)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "transcripts"), exist_ok=True)
    transcripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "transcripts"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(transcripts_dir, f"transcript_{timestamp}.txt")

    model = Model(args.model)
    rec = KaldiRecognizer(model, args.samplerate)
    rec.SetWords(True)  # include word timestamps if model supports it

    print("Using model:", args.model)
    print("Press Ctrl+C to stop. Listing microphone input...")

    try:
        with open(out_path, "a", encoding="utf-8") as fout:
            with sd.RawInputStream(samplerate=args.samplerate, blocksize=args.blocksize, dtype='int16',
                                   channels=1, callback=audio_callback, device=args.device):
                print("Listening (sample rate:", args.samplerate, ") ...")
                while True:
                    data = q.get()
                    if rec.AcceptWaveform(data):
                        res = json.loads(rec.Result())
                        text = res.get("text", "").strip()
                        if text:
                            print("\nFINAL:", text)
                            fout.write(text + "\n")
                            fout.flush()
                    else:
                        pres = json.loads(rec.PartialResult())
                        partial = pres.get("partial", "").strip()
                        # Print partial inline (overwrite line)
                        if partial:
                            print("PARTIAL: " + partial, end="\r")
    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        # On exit, get final result buffer and print/save
        final = json.loads(rec.FinalResult())
        final_text = final.get("text", "").strip()
        if final_text:
            print("FINAL (on exit):", final_text)
            with open(out_path, "a", encoding="utf-8") as fout:
                fout.write(final_text + "\n")
        print("Transcript saved to:", out_path)
    except Exception as e:
        print("Error:", str(e))
        raise

if __name__ == "__main__":
    main()