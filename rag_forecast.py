# rag_forecast.py
import re
import pandas as pd
import numpy as np
import chromadb
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from urllib.parse import quote_plus
from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
    CHROMA_DIR, CHROMA_COLLECTION, EMBED_MODEL
)

# DB + Vector Setup

engine = create_engine(
    f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4",
    pool_pre_ping=True, pool_recycle=1800
)

chroma = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma.get_or_create_collection(CHROMA_COLLECTION)
embedder = SentenceTransformer(EMBED_MODEL)

# =====
# Helpers
# =====
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

def _metrics_report(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    smape = float(100.0 * np.mean(
        2 * np.abs(y_true - y_pred) /
        np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)))
    ))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE%": round(mape, 2),
        "sMAPE%": round(smape, 2)
    }

# ==========================
# RAG
# ==========================
def query_rag(user_query: str, top_k: int = 5):
    q_emb = embedder.encode([user_query], convert_to_numpy=True).tolist()
    res = collection.query(
        query_embeddings=q_emb,
        n_results=top_k * 3,
        include=["documents", "distances"]
    )
    docs = (res.get("documents") or [[]])[0]

    clean_docs = []
    for d in docs:
        if not d:
            continue
        txt = d.strip()
        if len(txt) < 30:
            continue
        if re.fullmatch(r"[\d\W]+", txt):
            continue
        clean_docs.append(txt)

    return clean_docs[:top_k]

# ==========================
# Forecast Utilities
# ==========================
def infer_target(user_query: str) -> str:
    q = user_query.lower()
    if "premium" in q:
        return "premium"
    if "claim" in q:
        return "amount_claimed"
    return "premium"

def infer_horizon(user_query: str) -> int:
    q = user_query.lower()
    m = re.search(r"next\s+(\d+)\s*months?", q)
    if m: return int(m.group(1)) * 30
    m = re.search(r"next\s+(\d+)\s*years?", q)
    if m: return int(m.group(1)) * 365
    m = re.search(r"next\s+(\d+)\s*days?", q)
    if m: return int(m.group(1))
    return 180  # default ~6 months

def fetch_sql_data():
    sql = """
    SELECT
        start_date,
        end_date,
        premium,
        manufacturer,
        amount_claimed,
        claim_made,
        DATEDIFF(end_date, COALESCE(claim_date, start_date)) AS days_to_expire,
        COALESCE(claim_date, start_date) AS ts_date
    FROM policies
    ORDER BY ts_date ASC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)

def prepare_timeseries(df: pd.DataFrame, target: str):
    df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
    df = df.dropna(subset=["ts_date"])
    df = df.sort_values("ts_date")

    y = df[target].astype(float)
    X = pd.DataFrame({
        "days_to_expire": pd.to_numeric(df["days_to_expire"], errors="coerce").fillna(0),
        "claim_made": pd.to_numeric(df["claim_made"], errors="coerce").fillna(0)
    }, index=df["ts_date"])

    return X, y

def train_forecast(X, y, horizon_days: int):
    data = X.copy()
    data["y"] = y.values

    for lag in [1, 2, 7, 14, 30]:
        data[f"y_lag_{lag}"] = data["y"].shift(lag)

    data = data.dropna()
    y = data["y"]
    X = data.drop(columns=["y"])

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    return model, X, y

def recursive_forecast(model, X: pd.DataFrame, horizon_days: int):
    last_row = X.iloc[[-1]].copy()
    preds = []
    for step in range(horizon_days):
        y_hat = model.predict(last_row)[0]
        preds.append({"day": step + 1, "forecast": float(y_hat)})

        for lag in [1, 2, 7, 14, 30]:
            if lag == 1:
                last_row[f"y_lag_{lag}"] = y_hat
            else:
                last_row[f"y_lag_{lag}"] = last_row.get(f"y_lag_{lag-1}", y_hat)

    return pd.DataFrame(preds)

# ==========================
# Main Pipeline
# ==========================
def run_rag_forecast(user_query: str, history=None) -> dict:
    docs = query_rag(user_query, top_k=5)
    target = infer_target(user_query)
    horizon = infer_horizon(user_query)

    df = fetch_sql_data()
    if df.empty:
        return {"error": "No data in DB"}

    X, y = prepare_timeseries(df, target)
    if X.empty:
        return {"error": f"No usable data for target {target}"}

    if len(X) <= horizon + 20:
        return {"error": "Not enough history for validation"}
    X_train, X_test = X.iloc[:-horizon], X.iloc[-horizon:]
    y_train, y_test = y.iloc[:-horizon], y.iloc[-horizon:]

    model, X_full, y_full = train_forecast(X_train, y_train, horizon)
    y_backtest = model.predict(X_test)
    metrics = _metrics_report(y_test, y_backtest)

    fc = recursive_forecast(model, X_full, horizon)

    safe_fc = fc.astype(str).head(20)
    pipe_table = _df_to_pipe_table(safe_fc, max_rows=10)

    final_text = (
        f"**Forecast Target:** `{target}`\n\n"
        f"**Forecast Horizon:** {horizon} days\n\n"
        f"**Metrics:** {metrics}\n\n"
        f"**Forecast Results:**\n{pipe_table}"
    )

    return {
        "action": "rag_forecast",
        "user_query": user_query,
        "rag_context": docs,
        "forecast_target": target,
        "forecast_horizon_days": horizon,
        "metrics": metrics,
        "final_text": final_text,  # for CLI/debugging
        "table": {  # structured for Flask app rendering
            "columns": safe_fc.columns.tolist(),
            "rows": safe_fc.values.tolist()
        },
        "forecast_table": fc.to_dict(orient="records"),
        "dataframe": fc,
        "ModuleUsed": "RAG + Forecast"
    }

# ==========================
# CLI
# ==========================
if __name__ == "__main__":
    chat_history = []
    while True:
        q = input("\nEnter forecast query (or 'exit' to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        result = run_rag_forecast(q, history=chat_history)

        if "error" in result:
            print("\nError:", result["error"])
            continue

        print("\n--- RAG Context ---")
        for d in result.get("rag_context", []):
            print("-", d[:150], "...")

        print("\n--- Forecast ---")
        print(result["final_text"])

        chat_history.append({"sender": "User", "message": q})
        chat_history.append({"sender": "Bot", "message": result["final_text"]})






#############################





# # rag_forecast.py
# import re
# import pandas as pd
# import numpy as np
# import chromadb
# from sqlalchemy import create_engine, text
# from sentence_transformers import SentenceTransformer
# from xgboost import XGBRegressor
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
#     q_emb = embedder.encode([user_query], convert_to_numpy=True).tolist()
#     res = collection.query(
#         query_embeddings=q_emb,
#         n_results=top_k * 3,
#         include=["documents", "distances"]
#     )
#     docs = (res.get("documents") or [[]])[0]

#     # filter out junk (short/empty)
#     clean_docs = []
#     for d in docs:
#         if not d:
#             continue
#         txt = d.strip()
#         if len(txt) < 30:
#             continue
#         if re.fullmatch(r"[\d\W]+", txt):
#             continue
#         clean_docs.append(txt)

#     return clean_docs[:top_k]

# # ==========================
# # Forecast Utilities
# # ==========================
# def infer_target(user_query: str) -> str:
#     q = user_query.lower()
#     if "premium" in q:
#         return "premium"
#     if "claim" in q:
#         return "amount_claimed"
#     return "premium"

# def infer_horizon(user_query: str) -> int:
#     q = user_query.lower()
#     m = re.search(r"next\s+(\d+)\s*months?", q)
#     if m:
#         return int(m.group(1)) * 30
#     m = re.search(r"next\s+(\d+)\s*years?", q)
#     if m:
#         return int(m.group(1)) * 365
#     m = re.search(r"next\s+(\d+)\s*days?", q)
#     if m:
#         return int(m.group(1))
#     return 180  # default ~6 months

# def fetch_sql_data():
#     sql = """
#     SELECT
#         start_date,
#         end_date,
#         premium,
#         manufacturer,
#         amount_claimed,
#         claim_made,
#         DATEDIFF(end_date, COALESCE(claim_date, start_date)) AS days_to_expire,
#         COALESCE(claim_date, start_date) AS ts_date
#     FROM policies
#     ORDER BY ts_date ASC
#     """
#     with engine.connect() as conn:
#         return pd.read_sql(text(sql), conn)

# def prepare_timeseries(df: pd.DataFrame, target: str):
#     df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
#     df = df.dropna(subset=["ts_date"])
#     df = df.sort_values("ts_date")

#     y = df[target].astype(float)
#     X = pd.DataFrame({
#         "days_to_expire": pd.to_numeric(df["days_to_expire"], errors="coerce").fillna(0),
#         "claim_made": pd.to_numeric(df["claim_made"], errors="coerce").fillna(0)
#     }, index=df["ts_date"])

#     return X, y

# def train_forecast(X, y, horizon_days: int):
#     # build lag features
#     data = X.copy()
#     data["y"] = y.values

#     for lag in [1, 2, 7, 14, 30]:
#         data[f"y_lag_{lag}"] = data["y"].shift(lag)

#     data = data.dropna()
#     y = data["y"]
#     X = data.drop(columns=["y"])

#     # train
#     model = XGBRegressor(
#         n_estimators=200,
#         learning_rate=0.1,
#         max_depth=5,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(X, y)

#     # recursive forecast
#     last_row = X.iloc[[-1]].copy()
#     preds = []
#     for step in range(horizon_days):
#         y_hat = model.predict(last_row)[0]
#         preds.append({"day": step + 1, "forecast": float(y_hat)})

#         # shift lag values
#         for lag in [1, 2, 7, 14, 30]:
#             if lag == 1:
#                 last_row[f"y_lag_{lag}"] = y_hat
#             else:
#                 last_row[f"y_lag_{lag}"] = last_row.get(f"y_lag_{lag-1}", y_hat)

#     return pd.DataFrame(preds)

# # ==========================
# # Main Pipeline
# # ==========================
# def run_rag_forecast(user_query: str) -> dict:
#     docs = query_rag(user_query, top_k=5)
#     target = infer_target(user_query)
#     horizon = infer_horizon(user_query)

#     df = fetch_sql_data()
#     if df.empty:
#         return {"error": "No data in DB"}

#     X, y = prepare_timeseries(df, target)
#     if X.empty:
#         return {"error": f"No usable data for target {target}"}

#     fc = train_forecast(X, y, horizon)

#     return {
#         "user_query": user_query,
#         "rag_context": docs,
#         "forecast_target": target,
#         "forecast_horizon_days": horizon,
#         "forecast_table": fc.to_dict(orient="records")
#     }

# # ==========================
# # CLI
# # ==========================
# if __name__ == "__main__":
#     while True:
#         q = input("\nEnter forecast query (or 'exit' to quit): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         result = run_rag_forecast(q)

#         print("\n--- RAG Context ---")
#         for d in result.get("rag_context", []):
#             print("-", d[:150], "...")

#         if "error" in result:
#             print("\nError:", result["error"])
#             continue

#         print("\n--- Forecast Target ---")
#         print(result["forecast_target"])
#         print("\n--- Forecast Horizon (days) ---")
#         print(result["forecast_horizon_days"])
#         print("\n--- Forecast Result ---")
#         df = pd.DataFrame(result["forecast_table"])
#         print(df.head(15).to_string(index=False))
