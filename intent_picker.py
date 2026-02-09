# intent_picker.py
import json
import datetime
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Any

from config import GEMINI_API_KEY, GEMINI_MODEL

# Import the mode-specific modules
from rag_module import run_rag_query
from sql_module import run_sql_query
from forecast import run_forecast
from rag_sql import run_rag_sql
from rag_forecast import run_rag_forecast

IntentType = str

# ---------- JSON Safe Conversion ----------
def safe_json(obj):
    if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_json(x) for x in obj]
    return obj

# ---------- Intent Picker ----------
class IntentPicker:
    def __init__(self, debug: bool = True):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.conf_threshold = 0.75
        self.debug = debug

    # --- History formatter ---
    def _format_history(self, history: List) -> str:
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

        # """Format last turns of history for LLM context."""
        # if not history:
        #     return ""
        # convo = []
        # for sender, msg in history[-6:]:  # last 3 exchanges
        #     label = "User" if sender.lower().startswith("you") else "AI"
        #     convo.append(f"{label}: {msg}")
        # return "\n".join(convo)

        # """Format history items safely whether dicts or tuples."""
        # convo = []
        # for h in history:
        #     if isinstance(h, dict):
        #         role = h.get("sender", "User")
        #         msg = h.get("message", "")
        #     elif isinstance(h, tuple) and len(h) == 2:
        #         role, msg = h
        #     else:
        #         role, msg = "User", str(h)
        #     convo.append(f"{role}: {msg}")
        # return "\n".join(convo)

    # --- Heuristic classifier ---
    def heuristic_classify(self, query: str) -> IntentType:
        q = query.lower()

        if any(w in q for w in ["calculate", "forecast", "predict", "projection", "future", "next year", "project", "trend"]):
            if any(w in q for w in ["policy", "regulation", "clause", "tariff", "guideline", "law", "section"]):
                return "rag_forecast"
            return "forecast_only"

        if any(w in q for w in ["list", "count", "show", "average", "filter", "table", "rows",
                                "database", "sql", "query", "data", "summary", "breakdown"]):
            if any(w in q for w in ["policy", "regulation", "clause", "tariff", "guideline", "law", "section"]):
                return "rag_sql"
            return "sql_only"

        if any(w in q for w in [
            "policy", "regulation", "clause", "tariff", "guideline", "conditions",
            "motor tariff", "irdai", "insurance act", "law", "section", "rule",
            "coverage", "exclusion", "liability", "as per motor tariff"
        ]):
            return "rag_only"

        return "rag_only"

    # --- Gemini classifier ---
    def gemini_classify(self, query: str, history=None):
        hist_text = self._format_history(history)

        prompt = f"""
You are an expert assistant that classifies user queries for an Insurance AI system.
Consider the recent conversation:

{hist_text}

Classify the following query into exactly ONE intent label.
Return strict JSON with fields: "intent" and "confidence" (0-1).

Available intents:
- rag_only: Purely policy/regulation text (laws, clauses, tariffs, guidelines).
- sql_only: Purely database query (calculate, counts, averages, filters, lists, aggregations).
- forecast_only: Calculate, Prediction, projection, forecast, future trend.
- rag_sql: Needs BOTH policy/regulation text + database query.
- rag_forecast: Needs BOTH policy/regulation text + forecasting.

User query: "{query}"
"""
        try:
            resp = self.model.generate_content(prompt)
            txt = resp.text.strip()
            txt = txt.replace("```json", "").replace("```", "").strip()
            meta = json.loads(txt)

            intent = str(meta.get("intent", "")).lower()
            conf = float(meta.get("confidence", 0))

            if intent in ["rag_only", "sql_only", "forecast_only", "rag_sql", "rag_forecast"]:
                return intent, conf
        except Exception as e:
            if self.debug:
                print(f"[WARN] Gemini classify failed: {e}")

        return None, 0.0

    def classify(self, query: str, history=None) -> IntentType:
        llm_guess, conf = self.gemini_classify(query, history)

        if llm_guess and conf >= self.conf_threshold:
            final_intent = llm_guess
        else:
            final_intent = self.heuristic_classify(query)

        if final_intent not in ["rag_only", "sql_only", "forecast_only", "rag_sql", "rag_forecast"]:
            final_intent = "rag_only"

        if self.debug:
            print(f"[DEBUG] Query: {query}")
            print(f"[DEBUG] Gemini guess: {llm_guess}, conf={conf}")
            print(f"[DEBUG] Heuristic guess: {self.heuristic_classify(query)}")
            print(f"[DEBUG] Final intent: {final_intent}")

        return final_intent

    def route(self, query: str, history=None):
        intent = self.classify(query, history)

        if intent == "rag_only":
            return run_rag_query(query, history=history)
        elif intent == "sql_only":
            print("but hereeeeeeee")
            return run_sql_query(query, history=history)
        elif intent == "forecast_only":
            print("hiiiiiiiiiiiiiiiiiiii")
            return run_forecast(query, history=history)
        elif intent == "rag_sql":
            return run_rag_sql(query, history=history)
        elif intent == "rag_forecast":
            return run_rag_forecast(query, history=history)

        result = safe_json(result)

        mapping = {
            "rag_only": "RAG",
            "sql_only": "SQL",
            "forecast_only": "Forecast",
            "rag_sql": "RAG + SQL",
            "rag_forecast": "RAG + Forecast"
        }
        result["ModuleUsed"] = mapping[intent]

        return result

# ========= CLI ========= #
if __name__ == "__main__":
    picker = IntentPicker(debug=True)
    history = []
    while True:
        q = input("\nEnter query (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        history.append(("You", q))
        res = picker.route(q, history)
        print("\n--- Result ---")
        print(res)
        print("Module:", res.get("ModuleUsed"))
        history.append(("Bot", str(res)))







#######################


# # intent_picker.py
# import json
# import datetime
# import pandas as pd
# import google.generativeai as genai

# from config import GEMINI_API_KEY, GEMINI_MODEL

# # Import the mode-specific modules
# from rag_module import run_rag_query
# from sql_module import run_sql_query
# from forecast import run_forecast
# from rag_sql import run_rag_sql
# from rag_forecast import run_rag_forecast

# IntentType = str

# # ---------- JSON Safe Conversion ----------
# def safe_json(obj):
#     if isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
#         return obj.isoformat()
#     if isinstance(obj, dict):
#         return {k: safe_json(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [safe_json(x) for x in obj]
#     return obj

# # ---------- Intent Picker ----------
# class IntentPicker:
#     def __init__(self, debug: bool = True):
#         genai.configure(api_key=GEMINI_API_KEY)
#         self.model = genai.GenerativeModel(GEMINI_MODEL)
#         self.conf_threshold = 0.75   # confidence threshold for trusting Gemini
#         self.debug = debug

#     # --- History formatter ---
#     def format_history(self, history):
#         if not history:
#             return ""
#         convo = []
#         for sender, msg in history[-10:]:
#             label = "User" if sender == "You" else "AI"
#             convo.append(f"{label}: {msg}")
#         return "\n".join(convo)

#     # --- Heuristic classifier ---
#     def heuristic_classify(self, query: str) -> IntentType:
#         q = query.lower()

#         if any(w in q for w in ["forecast", "predict", "projection", "future", "next year", "project", "trend"]):
#             if any(w in q for w in ["policy", "regulation", "clause", "tariff", "guideline", "law", "section"]):
#                 return "rag_forecast"
#             return "forecast_only"

#         if any(w in q for w in ["list", "count", "show", "average", "filter", "table", "rows",
#                                 "database", "sql", "query", "data", "summary", "breakdown"]):
#             if any(w in q for w in ["policy", "regulation", "clause", "tariff", "guideline", "law", "section"]):
#                 return "rag_sql"
#             return "sql_only"

#         if any(w in q for w in [
#             "policy", "regulation", "clause", "tariff", "guideline", "conditions",
#             "motor tariff", "irdai", "insurance act", "law", "section", "rule",
#             "coverage", "exclusion", "liability", "premium rate"
#         ]):
#             return "rag_only"

#         # Default fallback
#         return "rag_only"

#     # --- Gemini classifier ---
#     def gemini_classify(self, query: str, history=None):
#         hist_text = self.format_history(history)

#         prompt = f"""
# You are an expert assistant that classifies user queries for an Insurance AI system. 
# Consider the conversation history for context:

# {hist_text}

# Now decide the ONE most suitable intent for the latest user query.

# Rules:
# 1. Always output valid JSON only, with fields: "intent" and "confidence" (0-1).
# 2. Be strict about intent categories, no free text.

# Available intents:
# - rag_only
# - sql_only
# - forecast_only
# - rag_sql
# - rag_forecast

# User query: "{query}"
# """
#         try:
#             resp = self.model.generate_content(prompt)
#             txt = resp.text.strip()
#             txt = txt.replace("```json", "").replace("```", "").strip()
#             meta = json.loads(txt)

#             intent = str(meta.get("intent", "")).lower()
#             conf = float(meta.get("confidence", 0))

#             if intent in ["rag_only", "sql_only", "forecast_only", "rag_sql", "rag_forecast"]:
#                 return intent, conf
#         except Exception as e:
#             if self.debug:
#                 print(f"[WARN] Gemini classify failed: {e}")

#         return None, 0.0

#     def classify(self, query: str, history=None) -> IntentType:
#         llm_guess, conf = self.gemini_classify(query, history)

#         if llm_guess and conf >= self.conf_threshold:
#             final_intent = llm_guess
#         else:
#             final_intent = self.heuristic_classify(query)

#         # Final guardrail: enforce one of the 5
#         if final_intent not in ["rag_only", "sql_only", "forecast_only", "rag_sql", "rag_forecast"]:
#             final_intent = "rag_only"

#         if self.debug:
#             print(f"[DEBUG] Query: {query}")
#             print(f"[DEBUG] Gemini guess: {llm_guess}, conf={conf}")
#             print(f"[DEBUG] Heuristic guess: {self.heuristic_classify(query)}")
#             print(f"[DEBUG] Final intent: {final_intent}")

#         return final_intent

#     def route(self, query: str, history=None):
#         intent = self.classify(query, history)

#         if intent == "rag_only":
#             result = run_rag_query(query)
#         elif intent == "sql_only":
#             result = run_sql_query(query)
#         elif intent == "forecast_only":
#             result = run_forecast(query)
#         elif intent == "rag_sql":
#             result = run_rag_sql(query, history)
#         elif intent == "rag_forecast":
#             result = run_rag_forecast(query, history)
#         else:
#             # Should never happen now
#             result = {"error": "Unexpected routing", "query": query}

#         result = safe_json(result)

#         # Always assign ModuleUsed deterministically
#         mapping = {
#             "rag_only": "RAG",
#             "sql_only": "SQL",
#             "forecast_only": "Forecast",
#             "rag_sql": "RAG + SQL",
#             "rag_forecast": "RAG + Forecast"
#         }
#         result["ModuleUsed"] = mapping[intent]

#         return result

# # ========= CLI ========= #
# if __name__ == "__main__":
#     picker = IntentPicker(debug=True)
#     history = []
#     while True:
#         q = input("\nEnter query (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         history.append(("You", q))
#         res = picker.route(q, history)
#         print("\n--- Result ---")
#         print(res)
#         print("Module:", res.get("ModuleUsed"))
#         history.append(("Bot", str(res)))






###########################


