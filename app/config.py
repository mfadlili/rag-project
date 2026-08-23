import os
from functools import lru_cache

from openai import OpenAI

DEFAULT_OPENAI_BASE_URL = "https://api.notispaces.cloud/v1"
DEFAULT_CHAT_MODEL = "notispace-v1"
DEFAULT_EMBEDDING_MODEL = "notispace/ns-embed"
DEFAULT_CHROMA_DIR = "./data/chroma"

CHAT_MODEL = os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
CHROMA_DIR = os.getenv("CHROMA_DIR", DEFAULT_CHROMA_DIR)


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    base_url = os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)