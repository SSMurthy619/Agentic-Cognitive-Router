# rag_module.py — Gemini RAG with strict, clean, professional answers (Flash-compatible)

import os
import re
from typing import Dict, Any, List

import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from config import GEMINI_API_KEY, GEMINI_MODEL, CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass

# ========= Setup ========= #
genai.configure(api_key=GEMINI_API_KEY)
chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBED_MODEL)

# ========= Prompt ========= #
PLAIN_PROMPT = """Your role is: You are an assistant that helps answer questions based on Motor Tariff Acts and other insurance-related policy documents. Your primary function is to answer questions strictly based on the provided documents, while behaving in a professional and human-like manner.
You have to make sure: If you see [TABLE_START] and [TABLE_END] in context, interpret the content as a structured table.
Retain the number of columns of the table retrived from the document as it is.
Preserve formatting when summarizing or quoting such sections.
 - In a table, render as columns with a simple ASCII layout:
     Period                    | Rate
     <row 1>                   | <value>
     <row 2>                   | <value>

Respond in below format:
1) Title (one line)
2) Summarize (2–4 concise sentences)
3) Key points (bulleted list, ≤ 8 bullets)

At the end of each response provide the reference as mentioned below.
--> References: comma-separated list of chunk ids (e.g., doc_12, doc_45)

Conversation context (last turns may be relevant):
{history_text}

USER QUESTION:
{user_req}

CONTEXT CHUNKS:
{context_text}
"""

# ========= Helpers ========= #
def _extract_text(resp) -> str:
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        if hasattr(resp, "candidates") and resp.candidates:
            parts = []
            for cand in resp.candidates:
                if hasattr(cand, "content") and cand.content:
                    for p in getattr(cand.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            parts.append(p.text.strip())
            if parts:
                return "\n".join(parts)
    except Exception:
        pass
    return str(resp)

def ask_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            [system_prompt, user_prompt],
            generation_config=genai.types.GenerationConfig(temperature=temperature)
        )
        return _clean_llm_text(_extract_text(resp))
    except Exception as e:
        return f"[Gemini Error: {e}]"

def _format_context(ids: List[str], docs: List[str]) -> str:
    blocks = []
    for cid, doc in zip(ids, docs):
        text = (doc or "").strip().replace("\n", " ")
        if len(text) > 900:
            text = text[:900] + " …"
        blocks.append(f"[{cid}] {text}")
    return "\n\n".join(blocks)

def _clean_llm_text(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"^```(?:\w+)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _format_history(history: List) -> str:
    """Format last 10 turns of history into plain text (handles dicts & tuples)."""
    if not history:
        return ""
    convo = []
    for h in history[-10:]:
        if isinstance(h, dict):
            sender = h.get("sender", "")
            msg = h.get("message", "")
        elif isinstance(h, tuple) and len(h) == 2:
            sender, msg = h
        else:
            continue

        label = "User" if sender in ("You", "User") else "AI"
        convo.append(f"{label}: {msg}")
    return "\n".join(convo)

# ========= Main RAG ========= #
def run_rag_query(query: str, top_k: int = 4, history: List = None) -> Dict[str, Any]:
    try:
        print(f"[DEBUG] New RAG query: {query}")
        q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
        print(f"[DEBUG] Embedding dim: {len(q_emb[0])}")

        res = collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        docs = (res.get("documents") or [[]])[0]
        ids = (res.get("ids") or [[]])[0] if res.get("ids") else [f"doc_{i+1}" for i in range(len(docs))]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]

        if not docs:
            return {
                "Category": "RAG Answer",
                "Why": "No relevant context retrieved from the vector DB.",
                "AnswerText": "Title: Insufficient Context\n\nSummary:\n- No relevant passages retrieved.\n\nKey points:\n- Try including policy type, clause, or section name.\n\nCitations:\nConfidence: low",
                "Sources": [],
                "ModuleUsed": "RAG"
            }

        context_text = _format_context(ids, docs)
        history_text = _format_history(history)

        prompt = PLAIN_PROMPT.format(
            user_req=query,
            context_text=context_text,
            history_text=history_text
        )

        answer = ask_gemini(
            "You are a helpful assistant that must answer strictly from provided context.",
            prompt,
            temperature=0.2,
        )

        previews = []
        for i, (cid, d, doc) in enumerate(zip(ids, dists, docs)):
            snip = (doc or "").strip().replace("\n", " ")
            if len(snip) > 160:
                snip = snip[:160] + " …"
            previews.append({
                "id": cid,
                "distance": float(d) if d is not None else None,
                "preview": snip,
                "metadata": metas[i] if i < len(metas) else {}
            })

        return {
            "Category": "RAG Answer",
            "Why": "Answered strictly from top-k retrieved chunks in Chroma.",
            "AnswerText": answer,
            "Sources": previews[:5],
            "ModuleUsed": "RAG"
        }

    except Exception as e:
        print(f"[DEBUG] Exception in run_rag_query: {e}")
        return {
            "Category": "RAG Answer",
            "Why": f"Error occurred: {e}",
            "AnswerText": "Could not process your RAG request due to an internal error.",
            "Sources": [],
            "ModuleUsed": "RAG"
        }

# ========= CLI ========= #
if __name__ == "__main__":
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        res = run_rag_query(q)
        print("\n=== Answer ===")
        print(res["AnswerText"])
        print("\nSources:", [s["id"] for s in res.get("Sources", [])])
        print("Module:", res.get("ModuleUsed"))





################


# # rag_module.py — Gemini RAG with strict, clean, professional answers (Flash-compatible)

# import os
# import re
# from typing import Dict, Any, List

# import chromadb
# from sentence_transformers import SentenceTransformer
# import google.generativeai as genai
# from config import GEMINI_API_KEY, GEMINI_MODEL, CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL

# try:
#     from dotenv import load_dotenv, find_dotenv
#     load_dotenv(find_dotenv(), override=True)
# except Exception:
#     pass

# # ========= Setup ========= #
# genai.configure(api_key=GEMINI_API_KEY)
# chroma = chromadb.PersistentClient(path=CHROMA_DIR)
# collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION)
# embedder = SentenceTransformer(EMBED_MODEL)

# # ========= Prompt ========= #
# PLAIN_PROMPT = """You are an assistant that helps answer questions based on Motor Tariff Acts and other insurance-related policy documents.
# If you see [TABLE_START] and [TABLE_END] in context, interpret the content as a structured table.
# Retain the number of columns of the table retrived from the document as it is.
# Preserve formatting when summarizing or quoting such sections.

# Use ONLY the context provided. If you do not know the answer, say you don't know and do not make up an answer.


# At the end of each response provide the reference as mentioned below.
# --> References: comma-separated list of chunk ids (e.g., doc_12, doc_45)

# USER QUESTION:
# {user_req}

# CONTEXT CHUNKS:
# {context_text}
# """
# #######

# ##If the context is insufficient, clearly say so in the Summary and set Confidence: low.

# #Format output EXACTLY as follows:

# #1) Title (one line)
# #2) Summary (2–4 concise sentences)
# #Summarize answers nicely in 3 sentences based on the context.
# #3) Key points (bulleted list, ≤ 8 bullets)
# # --> Confidence: high | medium | low
# #######

# # ========= Helpers ========= #
# def _extract_text(resp) -> str:
#     """Extract plain text from Gemini response, works for Flash & Pro."""
#     try:
#         if hasattr(resp, "text") and resp.text:
#             return resp.text.strip()
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = []
#             for cand in resp.candidates:
#                 if hasattr(cand, "content") and cand.content:
#                     for p in getattr(cand.content, "parts", []):
#                         if hasattr(p, "text") and p.text:
#                             parts.append(p.text.strip())
#             if parts:
#                 return "\n".join(parts)
#     except Exception:
#         pass
#     return str(resp)

# def ask_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
#     """Query Gemini (Flash or Pro) with robust handling."""
#     try:
#         model = genai.GenerativeModel(GEMINI_MODEL)
#         resp = model.generate_content(
#             [system_prompt, user_prompt],
#             generation_config=genai.types.GenerationConfig(temperature=temperature)
#         )
#         return _clean_llm_text(_extract_text(resp))
#     except Exception as e:
#         return f"[Gemini Error: {e}]"

# def _format_context(ids: List[str], docs: List[str]) -> str:
#     blocks = []
#     for cid, doc in zip(ids, docs):
#         text = (doc or "").strip().replace("\n", " ")
#         if len(text) > 900:
#             text = text[:900] + " …"
#         blocks.append(f"[{cid}] {text}")
#     return "\n\n".join(blocks)

# def _clean_llm_text(s: str) -> str:
#     t = (s or "").strip()
#     t = re.sub(r"^```(?:\w+)?\s*", "", t, flags=re.I)
#     t = re.sub(r"\s*```$", "", t)
#     t = re.sub(r"\n{3,}", "\n\n", t)
#     return t.strip()

# # ========= Main RAG ========= #
# def run_rag_query(query: str, top_k: int = 4) -> Dict[str, Any]:
#     try:
#         print(f"[DEBUG] New RAG query: {query}")
#         q_emb = embedder.encode([query], convert_to_numpy=True).tolist()
#         print(f"[DEBUG] Embedding dim: {len(q_emb[0])}")

#         res = collection.query(
#             query_embeddings=q_emb,
#             n_results=top_k,
#             include=["documents", "distances", "metadatas"]
#         )

#         docs = (res.get("documents") or [[]])[0]
#         ids = (res.get("ids") or [[]])[0] if res.get("ids") else [f"doc_{i+1}" for i in range(len(docs))]
#         dists = (res.get("distances") or [[]])[0]
#         metas = (res.get("metadatas") or [[]])[0]

#         if not docs:
#             return {
#                 "Category": "RAG Answer",
#                 "Why": "No relevant context retrieved from the vector DB.",
#                 "AnswerText": "Title: Insufficient Context\n\nSummary:\n- No relevant passages retrieved.\n\nKey points:\n- Try including policy type, clause, or section name.\n\nCitations:\nConfidence: low",
#                 "Sources": [],
#                 "ModuleUsed": "RAG"
#             }

#         context_text = _format_context(ids, docs)
#         prompt = PLAIN_PROMPT.format(user_req=query, context_text=context_text)

#         answer = ask_gemini(
#             "You are a helpful assistant that must answer strictly from provided context.",
#             prompt,
#             temperature=0.2,
#         )

#         previews = []
#         for i, (cid, d, doc) in enumerate(zip(ids, dists, docs)):
#             snip = (doc or "").strip().replace("\n", " ")
#             if len(snip) > 160:
#                 snip = snip[:160] + " …"
#             previews.append({
#                 "id": cid,
#                 "distance": float(d) if d is not None else None,
#                 "preview": snip,
#                 "metadata": metas[i] if i < len(metas) else {}
#             })

#         return {
#             "Category": "RAG Answer",
#             "Why": "Answered strictly from top-k retrieved chunks in Chroma.",
#             "AnswerText": answer,
#             "Sources": previews[:5],
#             "ModuleUsed": "RAG"
#         }

#     except Exception as e:
#         print(f"[DEBUG] Exception in run_rag_query: {e}")
#         return {
#             "Category": "RAG Answer",
#             "Why": f"Error occurred: {e}",
#             "AnswerText": "Could not process your RAG request due to an internal error.",
#             "Sources": [],
#             "ModuleUsed": "RAG"
#         }

# # ========= CLI ========= #
# if __name__ == "__main__":
#     while True:
#         q = input("\nEnter query (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         res = run_rag_query(q)
#         print("\n=== Answer ===")
#         print(res["AnswerText"])
#         print("\nSources:", [s["id"] for s in res.get("Sources", [])])
#         print("Module:", res.get("ModuleUsed"))
