import os, time
from openai import OpenAI

client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ["GEMINI_API_KEY"])
CHAT_MODEL, EMBED_MODEL = "gemini-2.5-flash", "gemini-embedding-001"

def embed(texts, retries=4):
    for i in range(retries):
        try:
            r = client.embeddings.create(model=EMBED_MODEL, input=texts)
            return [d.embedding for d in r.data]
        except Exception:
            time.sleep(2 ** i * 5)   # backoff for rate limits
    raise RuntimeError("Embedding failed")
