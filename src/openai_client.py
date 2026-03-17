"""
openai_client.py — Thin wrapper around OpenAI SDK for embeddings and chat completions.
"""

import os
import time
import numpy as np
from typing import List, Optional
from openai import OpenAI

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
        _client = OpenAI(api_key=api_key)
    return _client


def embed_texts(texts: List[str], model: str = "text-embedding-3-small", batch_size: int = 512) -> np.ndarray:
    """
    Embed a list of texts using the OpenAI Embeddings API.
    Returns a 2D numpy array of shape (len(texts), embedding_dim).
    Batches automatically and retries on rate-limit errors.
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        for attempt in range(5):
            try:
                response = client.embeddings.create(input=batch, model=model)
                batch_embs = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embs)
                break
            except Exception as e:
                if attempt < 4:
                    wait = 2 ** attempt
                    print(f"Embedding error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

    return np.array(all_embeddings, dtype=np.float32)


def embed_single(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a single string. Returns 1D numpy array."""
    return embed_texts([text], model=model)[0]


def chat_completion(
    prompt: str,
    system: str = "",
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """
    Single-turn chat completion. Returns the assistant message text.
    Set json_mode=True to request JSON output (model must be instructed to output JSON).
    """
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(5):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt
                print(f"Chat completion error (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
