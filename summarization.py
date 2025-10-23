
from typing import Optional
def summarize_text(text: str) -> str:
    """
    Summarize using transformers if available; fallback to extractive heuristics.
    """
    try:
        from transformers import pipeline
        summarizer = pipeline("summarization")
        # chunk if too long
        max_chunk = 1000
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        res = []
        for ch in chunks:
            out = summarizer(ch, max_length=150, min_length=40, do_sample=False)
            res.append(out[0]["summary_text"])
        return "\n\n".join(res)
    except Exception:
        # simple fallback: return first 3 sentences or first 300 chars
        import re
        sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sents) >= 3:
            return " ".join(sents[:3])
        return text[:500] + ("..." if len(text) > 500 else "")
