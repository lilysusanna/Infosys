
import os
from typing import List, Dict, Tuple

class DiarizationUnavailable(Exception):
    pass

def diarize_audio(audio_path: str) -> List[Dict]:
    """
    Attempt speaker diarization. Prefer pyannote if available.
    If unavailable, return a heuristic segmentation (fixed-length chunks)
    so the app can continue to function.
    """
    try:
        from pyannote.audio import Pipeline
        # try to load token from environment
        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        if token:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=token)
        else:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        diarization = pipeline(audio_path)
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)})
        if not segments:
            raise DiarizationUnavailable("pyannote returned no segments")
        return segments
    except Exception as e:
        # Fallback heuristic: split into 30s segments and label as Speaker 1/2 alternating
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            # last resort, assume 60 seconds
            duration = 60.0
        seg_len = 30.0
        segments = []
        t = 0.0
        sp = 0
        while t < duration:
            end = min(duration, t + seg_len)
            segments.append({"start": t, "end": end, "speaker": f"Speaker {sp+1}"})
            t = end
            sp = (sp + 1) % 2
        return segments

def assign_speakers_to_transcript(transcript: str, segments: List[Dict]) -> List[Dict]:
    """
    Very simple alignment: split transcript by sentences and assign to segments sequentially.
    This is a fallback when we don't have fine-grained alignment.
    """
    import re
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', transcript) if s.strip()]
    assigned = []
    si = 0
    for seg in segments:
        # join a couple of sentences per segment roughly
        if si >= len(sentences):
            break
        # take 1-3 sentences depending on remaining
        take = min(3, len(sentences)-si)
        text = " ".join(sentences[si:si+take])
        assigned.append({"start": seg["start"], "end": seg["end"], "speaker": seg["speaker"], "text": text})
        si += take
    # append any remaining sentences to last segment
    if si < len(sentences) and assigned:
        assigned[-1]["text"] += " " + " ".join(sentences[si:])
    elif si < len(sentences):
        # no segments, return whole transcript as single speaker
        assigned = [{"start": 0.0, "end": 0.0, "speaker": "Speaker 1", "text": transcript}]
    return assigned
