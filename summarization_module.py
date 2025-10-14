from typing import Optional
import os

# Try to use Google Generative (Gemini) if configured. Falls back to transformers t5-small.
_USE_GEMINI = False
_GEMINI_CLIENT = None
_GEMINI_MODEL = "gemini-1.5-flash"  # fast, low-latency default

# Global caches for faster summarization fallback
_HF_SUMMARIZER_PIPELINE = None
_HF_SUMMARIZER_TOKENIZER = None

try:
    import google.generativeai as genai  # type: ignore
    _api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GEMINI_API_KEY")
    if _api_key:
        genai.configure(api_key=_api_key)
        _USE_GEMINI = True
        _GEMINI_CLIENT = genai
except Exception:
    _USE_GEMINI = False


def _gemini_summarize(text: str, max_output_tokens: int = 256) -> str:
    """Summarize using Google Generative API (Gemini).

    Requires GOOGLE_API_KEY or GOOGLE_GEMINI_API_KEY in the environment and
    the `google-generative-ai` Python package available.
    """
    client = _GEMINI_CLIENT
    if client is None:
        raise RuntimeError("Gemini client not configured")

    prompt = (
        "You are a meeting summarization assistant. Produce a concise meeting summary "
        "(bullet points + short action items) while preserving speaker attributions. "
        "Input is a diarized transcript where each line begins with '[Speaker ...]:'.\n\n"
        "Meeting transcript:\n\n"
        f"{text}\n\n"
        "Return a compact summary with sections: Overview, Key points, Action items."
    )

    # Use the modern GenerativeModel API when available for speed and stability
    try:
        model = client.GenerativeModel(_GEMINI_MODEL)
        resp = model.generate_content(prompt, generation_config={"temperature": 0.2, "max_output_tokens": max_output_tokens})
        txt = getattr(resp, 'text', None)
        if not txt and hasattr(resp, 'candidates'):
            # Try harder to extract text
            for c in getattr(resp, 'candidates', []) or []:
                parts = getattr(getattr(c, 'content', None), 'parts', []) or []
                joined = "".join(getattr(p, 'text', '') for p in parts)
                if joined.strip():
                    txt = joined
                    break
        if not txt:
            txt = str(resp)
        return txt.strip()
    except Exception:
        # Legacy API fallback
        resp = client.generate_text(model=_GEMINI_MODEL, temperature=0.2, max_output_tokens=max_output_tokens, prompt=prompt)
        out = None
        try:
            out = getattr(resp, 'text', None)
        except Exception:
            out = None
        if not out:
            out = str(resp)
        return out.strip()


def _hf_fallback_summarize(text: str) -> str:
    """Fallback summarizer using transformers t5-small if Gemini isn't available."""
    global _HF_SUMMARIZER_PIPELINE, _HF_SUMMARIZER_TOKENIZER
    try:
        from transformers import pipeline, AutoTokenizer
    except Exception as e:
        raise RuntimeError("transformers not available and Gemini not configured") from e

    model_name = os.environ.get('SUMMARIZER_MODEL', 't5-small')
    device = os.environ.get('MODEL_DEVICE', None)
    if _HF_SUMMARIZER_TOKENIZER is None:
        _HF_SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    tokenizer = _HF_SUMMARIZER_TOKENIZER
    # pipeline will use GPU if device parameter is provided and supported
    pipeline_kwargs = {}
    if device:
        # transformers pipeline expects -1 for CPU, or int GPU index; try to map 'cuda' -> 0
        if device.startswith('cuda'):
            pipeline_device = 0
        else:
            pipeline_device = -1
        pipeline_kwargs['device'] = pipeline_device
    if _HF_SUMMARIZER_PIPELINE is None:
        _HF_SUMMARIZER_PIPELINE = pipeline("summarization", model=model_name, **pipeline_kwargs)
    summarizer = _HF_SUMMARIZER_PIPELINE

    # Transformers models often have a max input length (e.g., 512 tokens). If
    # the input is longer we split it into token chunks, summarize each, then
    # summarize the summaries.
    try:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        # fallback to naive split if tokenization fails
        tokenizers_max = 10000
        input_ids = tokenizer.encode(text[:tokenizers_max], add_special_tokens=False)

    max_input = getattr(tokenizer, "model_max_length", 512)
    # reserve some tokens for model internals / prompt
    chunk_size = max(128, max_input - 64)

    if len(input_ids) <= max_input:
        # single-pass summarization; use max_new_tokens to avoid the warning
        out = summarizer(text, max_new_tokens=128)
        return out[0]["summary_text"]

    # chunk by tokens
    chunks = []
    for i in range(0, len(input_ids), chunk_size):
        chunk_ids = input_ids[i : i + chunk_size]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)

    chunk_summaries = []
    for ch in chunks:
        try:
            s = summarizer(ch, max_new_tokens=128)
            chunk_summaries.append(s[0]["summary_text"])
        except Exception:
            # if a chunk fails, skip it
            continue

    combined = "\n".join(chunk_summaries)
    # final synthesis pass
    final = summarizer(combined, max_new_tokens=128)
    return final[0]["summary_text"]


def summarize_diarized_transcript(file_path: str) -> str:
    """Summarize a diarized transcript file (one line per speaker turn).

    Returns the summary string and writes it to <file_path>_summary.txt.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Parse diarized lines into per-speaker buckets. Expect lines like:
    # [Speaker SPEAKER_00]: some text
    speakers = {}
    encountered = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # naive parse: find first ']' after '['
        if line.startswith('[') and ']' in line and ':' in line:
            try:
                label = line[1:line.index(']')]
                # normalize label by removing leading 'Speaker ' if present
                label = label.replace('Speaker ', '').strip()
                content = line.split(':', 1)[1].strip()
            except Exception:
                label = 'UNKNOWN'
                content = line
        else:
            label = 'UNKNOWN'
            content = line

        speakers.setdefault(label, []).append(content)
        if label not in encountered:
            encountered.append(label)

    # Try to load a per-file speaker map if present. This allows mapping
    # model-generated labels to human names (e.g. SPEAKER_00 -> Alice).
    speaker_map = {}
    try:
        base = os.path.basename(file_path).replace('_diarized.txt', '')
        map_path = os.path.join('results', 'transcripts', f"{base}_speaker_map.json")
        if os.path.isfile(map_path):
            import json as _json
            with open(map_path, 'r', encoding='utf-8') as mf:
                speaker_map = _json.load(mf)
    except Exception:
        speaker_map = {}

    # Function to get human-friendly label
    def human_label(orig_label, index_map={}):
        # if mapping file provides a name, use it
        if orig_label in speaker_map:
            return speaker_map[orig_label]
        # common pattern: SPEAKER_00 or SPEAKER00 -> extract number
        import re
        m = re.search(r"(\d+)", orig_label)
        if m:
            num = int(m.group(1))
            return f"Speaker {num+1}"
        # fallback: assign incremental speaker numbers based on encounter order
        if orig_label not in index_map:
            index_map[orig_label] = len(index_map) + 1
        return f"Speaker {index_map[orig_label]}"

    per_summaries = {}
    # Summarize each speaker's contributions and keep a representative quote
    for label, turns in speakers.items():
        joined = '\n'.join(t for t in turns if t)
        if not joined.strip():
            per_summaries[label] = {'summary': '', 'quote': ''}
            continue

        # get a short representative quote (longest non-empty turn)
        quote = max((t for t in turns if t and t.strip()), key=len, default='')

        if _USE_GEMINI:
            prompt = (
                "Summarize the following speaker's contributions into 2-3 concise bullet points, "
                "and provide one representative short quote (<=30 words).\n\n"
                f"Speaker: {label}\n\n{joined}\n\n"
            )
            try:
                s = _gemini_summarize(prompt, max_output_tokens=180)
            except Exception:
                s = _hf_fallback_summarize(prompt)
        else:
            s = _hf_fallback_summarize("Summarize into 2-3 bullets and add one quote:\n\n" + joined)

        per_summaries[label] = {'summary': s, 'quote': quote}

    # Synthesize an overall summary that preserves speaker attributions
    synth_lines = ["Overview:"]
    # short overview using all per-speaker summaries concatenated
    combined = '\n'.join(f"Speaker {lbl}: {per_summaries[lbl]['summary']}" for lbl in per_summaries)
    if _USE_GEMINI:
        try:
            overall = _gemini_summarize("Synthesize a short meeting summary (3-5 bullets) from these per-speaker notes:\n\n" + combined, max_output_tokens=200)
        except Exception:
            overall = _hf_fallback_summarize("Synthesize a short meeting summary (3-5 bullets) from these per-speaker notes:\n\n" + combined)
    else:
        overall = _hf_fallback_summarize("Synthesize a short meeting summary (3-5 bullets) from these per-speaker notes:\n\n" + combined)

    # Build final text: overall + per-speaker bullets with quotes
    final = [overall, "\nPer-speaker notes:"]
    for lbl, data in per_summaries.items():
        final.append(f"Speaker {lbl} — Summary:\n{data['summary']}\nQuote: \"{data['quote']}\"")

    summary = '\n\n'.join(final)

    out_file = file_path.replace("_diarized.txt", "_summary.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Summary saved at: {out_file}")
    return summary


def summarize_text(input_data: Optional[str]):
    """Convenience wrapper used by the pipeline.

    Accepts either a file path (existing on disk) or a raw transcript string.
    Returns the summary text (string)."""
    # If caller passed a path to a diarized transcript file, reuse existing function
    if isinstance(input_data, str) and os.path.isfile(input_data):
        return summarize_diarized_transcript(input_data)

    # Otherwise assume input_data is raw text
    text = input_data or ""
    if not isinstance(text, str):
        text = str(text)

    # Basic cleanup to reduce noise and repeated filler tokens before summarization
    try:
        # Remove repeated single-letter tokens separated by spaces or punctuation
        import re
        text = re.sub(r"(?:\b[a-zA-Z]\b[\s.,;:!?-]*){3,}", " ", text)
        # Collapse multiple spaces/newlines
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    except Exception:
        pass

    if _USE_GEMINI:
        return _gemini_summarize(text)
    return _hf_fallback_summarize(text)

