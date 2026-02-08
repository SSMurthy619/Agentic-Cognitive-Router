# rag_sql.py
import re
import pandas as pd
import chromadb
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from urllib.parse import quote_plus
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
    CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL
)

# ==========================
# DB + Vector Setup
# ==========================
engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4",
    pool_pre_ping=True, pool_recycle=1800
)

chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBED_MODEL)

# ==========================
# Helpers
# ==========================
def _format_history(history):
    convo = []
    for h in history[-10:]:
        if isinstance(h, dict):  
            role = h.get("sender", "User")
            msg = h.get("message", "")
        elif isinstance(h, tuple) and len(h) == 2:  
            role, msg = h
        else:
            continue
        convo.append(f"{role}: {msg}")
    return "\n".join(convo)

def _df_to_pipe_table(df: pd.DataFrame, max_rows=10) -> str:
    if df.empty:
        return "(No rows)"
    df = df.head(max_rows)
    headers = " | ".join(df.columns.astype(str))
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join([headers, sep] + rows)

# ==========================
# RAG
# ==========================
def query_rag(user_query: str, top_k: int = 5):
    q_emb = embedder.encode([user_query], convert_to_numpy=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k,
        include=["documents", "distances"]
    )
    docs = (res.get("documents") or [[]])[0]
    return docs

# ==========================
# SQL Builder (rule-based)
# ==========================
def build_sql_from_context(user_query: str, rag_docs: list, history=None) -> str:
    q = user_query.lower()
    history_str = _format_history(history)

    # Basic heuristic: pick column
    if "premium" in q:
        select_col = "gross_premium"
    elif "claim" in q:
        select_col = "claim_amount"
    else:
        select_col = "*"

    where_clauses = []

    # Manufacturer filters
    for doc in rag_docs:
        d = doc.lower()
        if "honda" in q or "honda" in d:
            where_clauses.append("manufacturer LIKE '%Honda%'")
        if "toyota" in q or "toyota" in d:
            where_clauses.append("manufacturer LIKE '%Toyota%'")
        if "hyundai" in q or "hyundai" in d:
            where_clauses.append("manufacturer LIKE '%Hyundai%'")

    # Claims condition
    if "no claims" in q:
        where_clauses.append("claim_made_flag = '0'")
    elif "with claims" in q or "claims made" in q:
        where_clauses.append("claim_made_flag = '1'")

    # Days to expire filter
    m = re.search(r"days?\s*to\s*expire\s*(?:>|>=|more than|over)\s*(\d+)", q)
    if m:
        days = int(m.group(1))
        where_clauses.append(f"days_before_policy_expiry > {days}")

    # Country/location filter
    if "other than india" in q or "outside india" in q:
        where_clauses.append("(location <> 'India' AND location IS NOT NULL)")
    elif "india" in q:
        where_clauses.append("location = 'India'")

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
    sql = f"""
    SELECT {select_col}, policy_start_date, policy_end_date,
           manufacturer, claim_made_flag, claim_amount, state, zone
    FROM motorpolicies
    {where_sql}
    ORDER BY policy_start_date ASC
    LIMIT 50;
    """.strip()
    return sql

# ==========================
# SQL Runner
# ==========================
def run_sql(sql: str) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

# ==========================
# Main Pipeline
# ==========================
def run_rag_sql(user_query: str, history=None) -> dict:
    docs = query_rag(user_query, top_k=5)
    sql = build_sql_from_context(user_query, docs, history)
    df = run_sql(sql)

    safe_df = df.astype(str).head(20)
    pipe_table = _df_to_pipe_table(safe_df, max_rows=10)

    # Human-readable preview (for CLI)
    final_text = f"**SQL Query Used:**\n```\n{sql}\n```\n\n**Results:**\n{pipe_table}"

    return {
        "action": "rag_sql",
        "user_query": user_query,
        "rag_context": docs[:3],
        "sql": {"query": sql, "rows": len(df)},
        "final_text": final_text,  # for CLI/debugging
        "table": {
            "columns": safe_df.columns.tolist(),
            "rows": safe_df.values.tolist()
        },
        "dataframe": df,
        "ModuleUsed": "RAG + SQL"
    }

# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    chat_history = []
    while True:
        q = input("\nEnter query (or 'exit' to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = run_rag_sql(q, history=chat_history)

        print("\n--- RAG Context ---")
        for d in result["rag_context"]:
            print("-", d[:150], "...")

        print("\n--- SQL Query ---")
        print(result["sql"]["query"])

        print("\n--- SQL Result ---")
        print(result["final_text"])

        chat_history.append({"sender": "User", "message": q})
        chat_history.append({"sender": "Bot", "message": result["final_text"]})





##############################




# # rag_sql.py
# import re
# import pandas as pd
# import chromadb
# from sqlalchemy import create_engine, text
# from sentence_transformers import SentenceTransformer
# from urllib.parse import quote_plus
# from config import (
#     MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
#     CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL
# )

# # ==========================
# # DB + Vector Setup
# # ==========================
# engine = create_engine(
#     f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4",
#     pool_pre_ping=True, pool_recycle=1800
# )

# chroma = chromadb.PersistentClient(path=CHROMA_DIR)
# collection = chroma.get_or_create_collection(CHROMA_COLLECTION)
# embedder = SentenceTransformer(EMBED_MODEL)

# # ==========================
# # RAG
# # ==========================
# def query_rag(user_query: str, top_k: int = 5):
#     """Retrieve top-k docs from Chroma for guidance."""
#     q_emb = embedder.encode([user_query], convert_to_numpy=True).tolist()
#     res = collection.query(
#         query_embeddings=q_emb,
#         n_results=top_k,
#         include=["documents", "distances"]  # no ids anymore
#     )
#     docs = (res.get("documents") or [[]])[0]
#     return docs

# # ==========================
# # SQL Builder (rule-based)
# # ==========================
# def build_sql_from_context(user_query: str, rag_docs: list) -> str:
#     q = user_query.lower()

#     # Choose column(s)
#     if "premium" in q:
#         select_col = "premium"
#     elif "claim" in q:
#         select_col = "amount_claimed"
#     else:
#         select_col = "*"

#     where_clauses = []

#     # Manufacturer filters
#     for doc in rag_docs:
#         d = doc.lower()
#         if "honda" in q or "honda" in d:
#             where_clauses.append("manufacturer LIKE '%Honda%'")
#         if "toyota" in q or "toyota" in d:
#             where_clauses.append("manufacturer LIKE '%Toyota%'")
#         if "hyundai" in q or "hyundai" in d:
#             where_clauses.append("manufacturer LIKE '%Hyundai%'")

#     # Claims condition
#     if "no claims" in q:
#         where_clauses.append("claim_made = 0")
#     elif "with claims" in q or "claims made" in q:
#         where_clauses.append("claim_made = 1")

#     # Days to expire filter
#     m = re.search(r"days?\s*to\s*expire\s*(?:>|>=|more than|over)\s*(\d+)", q)
#     if m:
#         days = int(m.group(1))
#         where_clauses.append(f"DATEDIFF(end_date, COALESCE(claim_date, start_date)) > {days}")

#     # Country/location filter
#     if "other than india" in q or "outside india" in q:
#         where_clauses.append("(location <> 'India' AND location IS NOT NULL)")
#     elif "india" in q:
#         where_clauses.append("location = 'India'")

#     where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
#     sql = f"""
#     SELECT {select_col}, start_date, end_date, manufacturer, claim_made, amount_claimed, location
#     FROM policies
#     {where_sql}
#     LIMIT 50;
#     """.strip()
#     return sql

# # ==========================
# # SQL Runner
# # ==========================
# def run_sql(sql: str) -> pd.DataFrame:
#     with engine.connect() as conn:
#         return pd.read_sql(text(sql), conn)

# # ==========================
# # Main Pipeline
# # ==========================
# def run_rag_sql(user_query: str) -> dict:
#     docs = query_rag(user_query, top_k=5)
#     sql = build_sql_from_context(user_query, docs)
#     df = run_sql(sql)

#     return {
#         "user_query": user_query,
#         "rag_context": docs[:3],  # top docs only
#         "sql_query": sql,
#         "sql_result": df.to_dict(orient="records")
#     }

# # ==========================
# # CLI
# # ==========================
# if __name__ == "__main__":
#     while True:
#         q = input("\nEnter query (or 'exit' to quit): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         result = run_rag_sql(q)

#         print("\n--- RAG Context ---")
#         for d in result["rag_context"]:
#             print("-", d[:150], "...")

#         print("\n--- SQL Query ---")
#         print(result["sql_query"])

#         print("\n--- SQL Result ---")
#         df = pd.DataFrame(result["sql_result"])
#         print(df.head(10).to_string(index=False) if not df.empty else "No rows")
