from typing import List
import requests
from .config import OLLAMA_BASE_URL, EMBED_MODEL

def embed(text: str) -> List[float]:
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "text": text},
        timeout=120,
    )

    r.raise_for_status()
    data = r.json()
    if "embedding" not in data:
        raise ValueError(f"Embedding not found in response: {data}")
    return data["embedding"]