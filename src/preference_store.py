import json
import os
from datetime import datetime

PREF_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preferences.jsonl")

def save_preference(query: str, answer: str, label: str, retrieved_chunks: list = None):
    """
    Saves a user preference feedback entry.
    label: "positive" or "negative"
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "answer_preview": answer[:200],
        "chunk_texts": [c.get("text", "")[:300] for c in (retrieved_chunks or []) if isinstance(c, dict)],
        "label": label
    }
    # Ensure data directory exists
    os.makedirs(os.path.dirname(PREF_FILE), exist_ok=True)
    with open(PREF_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_preference_pairs():
    """
    Returns (pos_pairs, neg_pairs) for fine-tuning.
    Each pair is a tuple of (query, chunk_text).
    """
    if not os.path.exists(PREF_FILE):
        return [], []
    pos, neg = [], []
    with open(PREF_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                label = r.get("label", "positive")
                for text in r.get("chunk_texts", []):
                    pair = (r["query"], text)
                    if label == "positive":
                        pos.append(pair)
                    else:
                        neg.append(pair)
            except Exception:
                continue
    return pos, neg
