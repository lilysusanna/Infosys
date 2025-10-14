from pyannote.audio import Pipeline
import os
import json
import soundfile as sf
import numpy as np
import torch


def _iter_diarization_segments(diarization_output):
    """Yield (segment, speaker_label) pairs from a pipeline output.

    The pyannote speaker diarization pipeline can return either an
    Annotation (legacy behavior) or a DiarizeOutput dataclass with
    .speaker_diarization and .exclusive_speaker_diarization members.
    This helper normalizes both into an iterator of (segment, label).
    """
    from pyannote.core import Annotation

    if hasattr(diarization_output, "speaker_diarization"):
        annotation = diarization_output.speaker_diarization
    elif isinstance(diarization_output, Annotation):
        annotation = diarization_output
    else:
        raise TypeError("Unexpected diarization output type: %r" % type(diarization_output))

    for segment, _, label in annotation.itertracks(yield_label=True):
        yield segment, label

# Global cache for pyannote Pipeline to avoid reloading for every file
_DIARIZATION_PIPELINE = None

def diarize_audio(audio_file):
    """Run pyannote speaker diarization on an audio file.

    This function prefers passing audio as an in-memory waveform dict to the
    pyannote Pipeline to avoid depending on torchcodec/FFmpeg.
    """
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN")
    # optional device hint (e.g., 'cuda' or 'cpu')
    device_hint = os.environ.get('MODEL_DEVICE')
    if not hf_token:
        raise RuntimeError(
            "Hugging Face access token not found. Create a token at https://huggingface.co/settings/tokens "
            "(read access) and set it in your environment as HUGGING_FACE_HUB_TOKEN or HUGGING_FACE_TOKEN. "
            "You may also need to visit https://huggingface.co/pyannote/speaker-diarization and accept model terms."
        )

    # Use the explicit versioned pipeline and pass an explicit revision to avoid
    # downstream errors when Model.from_pretrained expects a `revision` kwarg.
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        _DIARIZATION_PIPELINE = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=hf_token, revision="main"
        )
    pipeline = _DIARIZATION_PIPELINE
    # Try to move pipeline to GPU if a CUDA device was requested
    try:
        device_hint = device_hint or os.environ.get('MODEL_DEVICE')
        if device_hint and device_hint.startswith('cuda'):
            # many pyannote objects support .to(device)
            try:
                pipeline.to(device_hint)
            except Exception:
                # ignore if not supported
                pass
    except Exception:
        pass

    # Load the file and pass audio in-memory to avoid torchcodec dependency.
    try:
        data, sr = sf.read(audio_file, dtype="float32")
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file '{audio_file}': {e}") from e

    # Convert to (channels, time) torch.Tensor of shape (channel, time)
    if data.ndim == 1:
        waveform = torch.from_numpy(data).unsqueeze(0)
    else:
        waveform = torch.from_numpy(data.T)

    diarization = pipeline({"waveform": waveform, "sample_rate": int(sr)})

    # Prefer the detailed whisper JSON (contains timestamps). If present,
    # align diarization segments to whisper segments by time overlap.
    base = os.path.basename(audio_file).replace('.wav', '')
    stt_json_path = os.path.join('results', 'transcripts', f"{base}_stt.json")
    stt_segments = []
    full_transcript_text = ""
    if os.path.isfile(stt_json_path):
        try:
            with open(stt_json_path, 'r', encoding='utf-8') as jf:
                stt_res = json.load(jf)
            full_transcript_text = stt_res.get('text', '') if isinstance(stt_res, dict) else ''
            # whisper returns a list of segments with start/end/text
            if isinstance(stt_res, dict) and 'segments' in stt_res:
                for s in stt_res['segments']:
                    # normalize keys (start, end, text)
                    start = float(s.get('start', 0.0))
                    end = float(s.get('end', start))
                    text = s.get('text', '').strip()
                    stt_segments.append({'start': start, 'end': end, 'text': text})
        except Exception:
            stt_segments = []
    else:
        # fallback: try reading plain txt
        stt_txt_path = os.path.join('results', 'transcripts', f"{base}_stt.txt")
        if os.path.isfile(stt_txt_path):
            try:
                with open(stt_txt_path, 'r', encoding='utf-8') as tf:
                    full_transcript_text = tf.read()
            except Exception:
                full_transcript_text = ''

    raw_turns = []
    for segment, speaker in _iter_diarization_segments(diarization):
        # find overlapping stt segments
        seg_start = float(segment.start)
        seg_end = float(segment.end)
        parts = []
        for s in stt_segments:
            # overlap if s.start < seg_end and s.end > seg_start
            if s['start'] < seg_end and s['end'] > seg_start:
                parts.append(s['text'])

        if not parts and stt_segments:
            # no overlap found; try nearest segment by midpoint
            midpoint = (seg_start + seg_end) / 2.0
            nearest = min(stt_segments, key=lambda x: abs((x['start'] + x['end'])/2.0 - midpoint))
            parts = [nearest['text']] if nearest and nearest.get('text') else []

        if not parts:
            # very degraded fallback: use full transcript (not ideal)
            text = full_transcript_text
        else:
            text = ' '.join(p for p in parts if p)

        # normalize whitespace, collapse repeated spaces/newlines
        text = ' '.join(text.split()) if isinstance(text, str) else ''

        raw_turns.append({'speaker': speaker, 'start': seg_start, 'end': seg_end, 'text': text})

    # Post-process turns: merge consecutive turns from same speaker,
    # remove exact/substring duplicates and collapse short repeated fragments.
    merged = []
    for turn in raw_turns:
        if not turn['text']:
            continue
        if merged and merged[-1]['speaker'] == turn['speaker']:
            # merge text and extend end time
            prev = merged[-1]
            combined = prev['text'] + ' ' + turn['text']
            # dedupe repeated substrings: if one text endswith the other's start, collapse
            # simple heuristic: if the second text is substring of the combined twice, trim
            # keep only one occurrence of any immediate repeated sequence
            if prev['text'].endswith(turn['text']):
                combined = prev['text']
            else:
                # remove overlapping prefix
                if combined.count(turn['text']) > 1:
                    # keep a single occurrence
                    combined = combined.replace(' ' + turn['text'], '', 1)

            prev['text'] = ' '.join(combined.split())
            prev['end'] = max(prev['end'], turn['end'])
        else:
            merged.append(turn.copy())

    # Remove exact duplicates and consecutive near-duplicates
    diarized_transcript = []
    seen_texts = set()
    for t in merged:
        txt = t['text'].strip()
        if not txt:
            continue
        # skip exact duplicates
        if txt in seen_texts:
            continue
        # skip if txt is substring of any previously kept text (avoid repeats)
        if any(txt in prev for prev in seen_texts):
            continue
        seen_texts.add(txt)
        diarized_transcript.append(f"[Speaker {t['speaker']}]: {txt}")

    out_file = f"results/transcripts/{os.path.basename(audio_file).replace('.wav','')}_diarized.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for line in diarized_transcript:
            f.write(line + "\n")

    # Also export RTTM for external tools/evaluation
    try:
        rttm_dir = os.path.join('results', 'diarization')
        os.makedirs(rttm_dir, exist_ok=True)
        base = os.path.basename(audio_file).replace('.wav','')
        rttm_path = os.path.join(rttm_dir, f"{base}.rttm")
        with open(rttm_path, 'w', encoding='utf-8') as rf:
            diarization.write_rttm(rf)
    except Exception:
        pass

    print(f"Diarized transcript saved at: {out_file}")
    return diarized_transcript
