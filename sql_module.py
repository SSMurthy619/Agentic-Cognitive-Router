import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from urllib.parse import quote_plus

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
    GEMINI_API_KEY, GEMINI_MODEL
)
import google.generativeai as genai

# ========= Setup ========= #
def get_engine() -> Engine:
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)

ENGINE = get_engine()
  
# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)

# ========= Explicit Schema ========= #
MOTORPOLICIES_SCHEMA = """
Table: motorpolicies
Columns:
- policy_number (varchar, primary key)
- policy_start_date (date)
- policy_end_date (date)
- manufacturer (varchar)
- model (varchar)
- zone (varchar)
- cc (int)
- state (varchar)
- city_tier (varchar)
- fuel_type (varchar)
- vehicle_age (int)
- zero_dep_flag (varchar)
- claim_made_flag (varchar)
- business_type (varchar)
- claim_intimation_date (date)
- days_to_claim_policy_start (int)
- days_before_policy_expiry (int)
- claim_amount (decimal)
- approved_amount (decimal)
- gross_premium (decimal)
- sum_insured_amount (decimal)
- claim_amt_idv_ratio (decimal)
- fraud_flag (varchar)
"""

# ========= Helpers ========= #
def clean_sql(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    t = re.sub(r"(?is)^\s*sql\s*:\s*", "", t)
    return t.strip()

def validate_select(sql: str):
    s = sql.strip().lower()
    if not re.match(r'^\s*(select|with)\b', s):
        raise ValueError("Query must start with SELECT or WITH")
    if re.search(r'\b(insert|update|delete|drop|alter|create|truncate)\b', s):
        raise ValueError("Only read-only queries allowed")

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
    """Convert a DataFrame to a Markdown pipe table (for HTML rendering)."""
    if df.empty:
        return "(No rows)"
    df = df.head(max_rows)
    headers = " | ".join(df.columns.astype(str))
    sep = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join([headers, sep] + rows)

# ========= Core ========= #
def generate_sql(user_input: str, table: str = "motorpolicies", history=None) -> str:
    # 🔹 Handle "loss ratio" explicitly
    if "loss ratio" in user_input.lower():
        return f"""
        SELECT EXTRACT(YEAR FROM policy_start_date) AS policy_year,
        SUM(claim_amount) AS total_loss_amount,
        SUM(gross_premium) AS total_premium,
        (SUM(claim_amount) / SUM(gross_premium)) AS LossRatio
        FROM {table}
        GROUP BY EXTRACT(YEAR FROM policy_start_date)
        
        """
    #ORDER BY ts_date ASC
    history_str = _format_history(history) if history else "No prior history."

    prompt = f"""
You are a MySQL assistant. Convert the user request into ONE safe SELECT.

Conversation so far:
{history_str}

Schema:
{MOTORPOLICIES_SCHEMA}

Rules:
- Use only listed columns.
- For "other than X / outside X": filter column <> 'X' and include NULL unless user says exclude nulls.
- Date columns available: policy_start_date, policy_end_date, claim_date.
- Premium-related analysis should use gross_premium.
- For IDV-related queries use idv or claim_amt_idv.
- Never invent columns. Never add LIMIT unless explicitly asked.
- Output only the SQL.
- Output table should always be printed with appropriate tabular view
- Display the SQL query used for retrieving

User: {user_input}
"""
    resp = model.generate_content(prompt)
    sql = clean_sql(resp.text)
    validate_select(sql)
    return sql

def run_sql_query(user_input: str, table: str = "motorpolicies", preview_n: int = 50, history=None) -> dict:
    sql = generate_sql(user_input, table, history)
    df = pd.read_sql(text(sql), ENGINE)

    safe_df = df.astype(str).head(preview_n)

    # Pipe-delimited table for rendering in chat
    pipe_table = _df_to_pipe_table(safe_df, max_rows=10)

    final_text = f"**SQL Query Used:**\n```\n{sql}\n```\n\n**Results:**\n{pipe_table}"

    return {
        "action": "sql_only",
        "sql": {"query": sql, "rows": len(df)},
        "final_text": final_text,
        "table": {
            "columns": safe_df.columns.tolist(),
            "rows": safe_df.values.tolist()
        },
        "dataframe": df
    }

# ========= CLI ========= #
if __name__ == "__main__":
    chat_history = []
    while True:
        q = input("Ask (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            res = run_sql_query(q, history=chat_history)
            print("\nSQL:\n", res["sql"]["query"])
            print("\nPreview:\n", res["final_text"])
            chat_history.append({"sender": "User", "message": q})
            chat_history.append({"sender": "Bot", "message": res["final_text"]})
        except Exception as e:
            print("Error:", e)




###################################

# import re
# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.engine import Engine
# from urllib.parse import quote_plus

# from config import (
#     MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
#     GEMINI_API_KEY, GEMINI_MODEL
# )
# import google.generativeai as genai

# # ========= Setup ========= #
# def get_engine() -> Engine:
#     url = (
#         f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
#         f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
#     )
#     return create_engine(url, pool_pre_ping=True)

# ENGINE = get_engine()

# # Gemini setup
# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel(GEMINI_MODEL)

# # ========= Explicit Schema ========= #
# MOTORPOLICIES_SCHEMA = """
# Table: motorpolicies
# Columns:
# - policy_number (varchar, primary key)
# - policy_start_date (date)
# - policy_end_date (date)
# - manufacturer (varchar)
# - model (varchar)
# - zone (varchar)
# - state (varchar)
# - city_tier (varchar)
# - fuel_type (varchar)
# - vehicle_age (int)
# - zero_dep_flag (varchar)
# - claim_made_flag (varchar)
# - business_type (varchar)
# - claim_intimation_date (date)
# - days_to_claim_policy_start (int)
# - days_before_policy_expiry (int)
# - claim_amount (decimal)
# - approved_amount (decimal)
# - gross_premium (decimal)
# - sum_insured_amount (decimal)
# - claim_amt_idv_ratio (decimal)
# - fraud_flag (varchar)
# """

# # ========= Helpers ========= #
# def clean_sql(s: str) -> str:
#     t = (s or "").strip()
#     t = re.sub(r"^```(?:sql)?\s*", "", t, flags=re.I)
#     t = re.sub(r"\s*```$", "", t)
#     t = re.sub(r"(?is)^\s*sql\s*:\s*", "", t)
#     return t.strip()

# def validate_select(sql: str):
#     s = sql.strip().lower()
#     if not re.match(r'^\s*(select|with)\b', s):
#         raise ValueError("Query must start with SELECT or WITH")
#     if re.search(r'\b(insert|update|delete|drop|alter|create|truncate)\b', s):
#         raise ValueError("Only read-only queries allowed")

# # ========= Core ========= #
# def generate_sql(user_input: str, table: str = "motorpolicies") -> str:
#     prompt = f"""
# You are a MySQL assistant. Convert the user request into ONE safe SELECT.

# Schema:
# {MOTORPOLICIES_SCHEMA}

# Rules:
# - Use only listed columns.
# - For "other than X / outside X": filter column <> 'X' and include NULL unless user says exclude nulls.
# - Date columns available: policy_start_date, policy_end_date, claim_date.
# - Premium-related analysis should use gross_premium.
# - For IDV-related queries use idv or claim_amt_idv.
# - Never invent columns. Never add LIMIT unless explicitly asked.
# - Output only the SQL.
# - Output table should always be printed with approprite tabular view
# - Display the SQL query used for retrieving

# User: {user_input}
# """

#     resp = model.generate_content(prompt)
#     sql = clean_sql(resp.text)
#     validate_select(sql)
#     return sql

# def run_sql_query(user_input: str, table: str = "motorpolicies", preview_n: int = 50) -> dict:
#     sql = generate_sql(user_input, table)
#     df = pd.read_sql(text(sql), ENGINE)

#     # console preview
#     def preview(df, n):
#         if df.empty:
#             return "(No rows)"
#         with pd.option_context("display.max_rows", n, "display.width", 140):
#             return df.head(n).to_string(index=False)

#     # JSON-safe table (convert dates/numbers to str for app)
#     safe_df = df.astype(str).head(preview_n)

#     return {
#         "action": "sql_only",

#         # SQL metadata
#         "sql": {"query": sql, "rows": len(df)},

#         # Human preview (console)
#         "final_text": preview(df, preview_n),

#         # JSON-safe version for app
#         "table": {
#             "columns": safe_df.columns.tolist(),
#             "rows": safe_df.values.tolist()
#         },

#         # Full DataFrame for CSV/export if needed
#         "dataframe": df
#     }

# # ========= CLI ========= #
# if __name__ == "__main__":
#     while True:
#         q = input("Ask (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         try:
#             res = run_sql_query(q)
#             print("\nSQL:\n", res["sql"]["query"])
#             print("\nPreview:\n", res["final_text"])
#         except Exception as e:
#             print("Error:", e)
