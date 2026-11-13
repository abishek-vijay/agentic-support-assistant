import chromadb, json, time
from config import embed

col = chromadb.PersistentClient("db").get_collection("rag-docs")
MAX_DIST = 0.45   # tune using your eval; lower = stricter

def search_knowledge_base(query):
    r = col.query(query_embeddings=embed([query]), n_results=3)
    hits = [{"source": m["source"], "text": d}
            for d, m, s in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])
            if s <= MAX_DIST]
    return hits or {"note": "No relevant documents found"}

ORDERS = {"1001": {"status": "Shipped", "eta": "2026-11-05"}}
def get_order_status(order_id): return ORDERS.get(order_id, {"error": "Order not found"})

def escalate_to_human(reason, summary):
    t = {"ticket": f"T-{int(time.time())}", "reason": reason, "summary": summary}
    open("tickets.jsonl", "a").write(json.dumps(t) + "\n")
    return {"ticket": t["ticket"], "message": "A human will follow up."}

def _t(name, desc, props):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": {k: {"type": "string"} for k in props},
                           "required": props}}}
TOOLS = [
  _t("search_knowledge_base", "Search company docs. Use for any policy or product question.", ["query"]),
  _t("get_order_status", "Get order status by ID.", ["order_id"]),
  _t("escalate_to_human", "Use if docs lack the answer, user is upset, or it involves refunds/account changes.", ["reason", "summary"]),
]
FUNCS = {"search_knowledge_base": search_knowledge_base,
         "get_order_status": get_order_status, "escalate_to_human": escalate_to_human}
