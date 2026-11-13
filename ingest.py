import pathlib, chromadb
from pypdf import PdfReader
from docx import Document
from config import embed

col = chromadb.PersistentClient("db").get_or_create_collection(
    "rag-docs", metadata={"hnsw:space": "cosine"})

def read(p):
    if p.suffix == ".pdf":  return "\n".join(pg.extract_text() or "" for pg in PdfReader(p).pages)
    if p.suffix == ".docx": return "\n".join(x.text for x in Document(p).paragraphs)
    return p.read_text(errors="ignore")

def chunk(t, size=2000, overlap=200):
    return [c for i in range(0, len(t), size - overlap)
            if len(c := t[i:i + size].strip()) > 20]

for p in pathlib.Path("docs").iterdir():
    if p.suffix not in {".pdf", ".docx", ".txt", ".md"}: continue
    chunks = chunk(read(p))
    for i in range(0, len(chunks), 20):
        b = chunks[i:i + 20]
        col.upsert(ids=[f"{p.name}-{i+j}" for j in range(len(b))],   # stable IDs, no duplicates
                   documents=b, embeddings=embed(b),
                   metadatas=[{"source": p.name, "chunk": i + j} for j in range(len(b))])
    print(p.name, len(chunks), "chunks")
