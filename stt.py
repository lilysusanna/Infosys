from typing import List, Tuple, Dict
from faster_whisper import WhisperModel
import torch


def transcribe_audio(audio_path: str, model_size: str = "small") -> Tuple[List[Dict], str]:
    """Transcribe audio file using faster-whisper.

    Returns list of segments and full text.
    """
    import os
    print(f"DEBUG: Starting transcription of {audio_path}")
    print(f"DEBUG: File exists: {os.path.exists(audio_path)}")
    print(f"DEBUG: File size: {os.path.getsize(audio_path) if os.path.exists(audio_path) else 'N/A'} bytes")

    # Auto-detect device and compute type
    if torch.cuda.is_available():
        compute_type = "float16"
        device = "cuda"
        print("DEBUG: Using CUDA")
    else:
        compute_type = "int8"
        device = "cpu"
        print("DEBUG: Using CPU")

    print(f"DEBUG: Loading Whisper model: {model_size}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("DEBUG: Model loaded successfully")
    except Exception as e:
        print(f"DEBUG: Error loading model: {e}")
        raise

    results: List[Dict] = []
    full_text_parts: List[str] = []

    try:
        print("DEBUG: Attempting transcription with VAD...")
        # First attempt with VAD
        segments, _ = model.transcribe(audio_path, vad_filter=True)
        segments = list(segments)
        print(f"DEBUG: VAD transcription returned {len(segments)} segments")
        if not segments:
            raise ValueError("No segments detected with VAD.")
    except Exception as e:
        print(f"DEBUG: VAD failed: {e}")
        print("DEBUG: Trying fallback without VAD...")
        try:
            # Fallback: no VAD + force English
            segments, _ = model.transcribe(audio_path, vad_filter=False, language="en")
            segments = list(segments)
            print(f"DEBUG: Fallback transcription returned {len(segments)} segments")
        except Exception as e2:
            print(f"DEBUG: Fallback also failed: {e2}")
            # Last resort: try without language specification
            try:
                segments, _ = model.transcribe(audio_path, vad_filter=False)
                segments = list(segments)
                print(f"DEBUG: No-language transcription returned {len(segments)} segments")
            except Exception as e3:
                print(f"DEBUG: All transcription attempts failed: {e3}")
                raise e3

    print(f"DEBUG: Processing {len(segments)} segments")
    for i, seg in enumerate(segments):
        print(f"DEBUG: Segment {i}: {seg.start:.2f}-{seg.end:.2f}s: '{seg.text.strip()}'")
        item = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        results.append(item)
        full_text_parts.append(item["text"])

    full_text = "\n".join(full_text_parts)
    print(f"DEBUG: Final result: {len(results)} segments, {len(full_text)} characters")
    print(f"DEBUG: Full text preview: '{full_text[:100]}...'")
    return results, full_text
