# text_utils.py
import re
from typing import List

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def split_sentences(text: str) -> List[str]:
    """Split a long speech into sentences (simple regex-based)."""
    if not text:
        return []
    text = text.replace("\n", " ").strip()
    parts = _SENT_SPLIT.split(text)
    # remove tiny/empty sentences
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) >= 2]
