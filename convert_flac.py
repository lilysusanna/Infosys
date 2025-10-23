import os
import subprocess
import argparse

def convert_flac_to_wav(input_dir, output_dir):
    """
    Converts all FLAC files in a directory to 16kHz mono WAV files.
    
    Args:
        input_dir (str): The directory containing FLAC files.
        output_dir (str): The directory to save the converted WAV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".flac"):
                flac_path = os.path.join(root, file)
                
                # Create corresponding output directory structure
                relative_path = os.path.relpath(flac_path, input_dir)
                wav_path_base = os.path.splitext(relative_path)[0]
                output_wav_path = os.path.join(output_dir, wav_path_base + ".wav")
                
                output_wav_dir = os.path.dirname(output_wav_path)
                if not os.path.exists(output_wav_dir):
                    os.makedirs(output_wav_dir)

                print(f"Converting {flac_path}...")
                
                try:
                    # ffmpeg command to convert to 16kHz, mono, 16-bit PCM WAV
                    command = [
                        'ffmpeg',
                        '-i', flac_path,
                        '-ac', '1',
                        '-ar', '16000',
                        '-sample_fmt', 's16',
                        '-y',  # Overwrite output file without asking
                        output_wav_path
                    ]
                    subprocess.run(command, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"Error converting {flac_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FLAC files to 16kHz mono WAV.")
    parser.add_argument("--input", required=True, help="Input directory containing FLAC files.")
    parser.add_argument("--output", required=True, help="Output directory for WAV files.")
    args = parser.parse_args()
    
    convert_flac_to_wav(args.input, args.output)
    print("\nBatch conversion complete.")