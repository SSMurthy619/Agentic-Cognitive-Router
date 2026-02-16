# forecast.py
  
import os
import re
import json
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import google.generativeai as genai

from config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
    GEMINI_API_KEY, GEMINI_MODEL
)

# ========= DB Engine ========= #
def get_engine():
    url = (
        f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
        f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)

ENGINE = get_engine()

# ========= Gemini Setup ========= #
genai.configure(api_key=GEMINI_API_KEY)
_gemini = genai.GenerativeModel(GEMINI_MODEL)

def _safe_extract_text(resp) -> str:
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        if hasattr(resp, "candidates") and resp.candidates:
            parts = []
            for c in resp.candidates:
                if hasattr(c, "content") and c.content:
                    for p in getattr(c.content, "parts", []):
                        if hasattr(p, "text") and p.text:
                            parts.append(p.text.strip())
            if parts:
                return "\n".join(parts)
    except Exception:
        pass
    return ""

# ========= Schema ========= #
NUMERIC_COLS = {
    "gross_premium",
    "claim_amount",
    "approved_amount",
    "sum_insured_amount",
    "vehicle_age",
    "cc",
    "days_to_claim_policy_start",
    "days_before_policy_expiry",
    "claim_amt_idv_ratio",
    # 🔹 synthetic
    "loss_ratio"
}

CATEGORICAL_COLS = {
    "policy_number","manufacturer","model","zone","state","city_tier",
    "fuel_type","zero_dep_flag","claim_made_flag","business_type","fraud_flag"
}

DATE_COLS_ALLOWED = {"policy_start_date","policy_end_date","claim_intimation_date"}
DEFAULT_DATE_COL = "policy_start_date"

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

NEG_SYNONYMS = {"no","n","false","0","non","non-zero dep","non zero dep","nonzerodep","without","exclude","excluding"}
POS_SYNONYMS = {"yes","y","true","1","zero dep","zerodep","fraud","with","include","including"}

# ========= Helpers ========= #
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

# ========= Metadata Inference ========= #
def infer_forecast_metadata(user_query: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    history_str = _format_history(history) if history else "No prior history."

    prompt = f"""
You are a Data Science forecasting assistant.

Conversation so far:
{history_str}

Now analyze the latest query.

Return strictly valid JSON with keys:
- target
- predictors
- filters
- horizon_days
- freq
- date_col

Rules:
1. Do NOT invent or assume columns unless the user explicitly mentions them.
2. If user does not mention a numeric predictor, leave "predictors": [].
3. If user does not mention filters, leave "filters": {{}}.
4. Always keep "target" limited to one numeric column (claim_amount, gross_premium, approved_amount, sum_insured_amount, loss_ratio, etc.).
5. Default "date_col" = "policy_start_date" unless the user explicitly names another.
6. Convert months/years to days for "horizon_days".
7. "freq" must be one of "D", "M", "Y". If not clear, default = "D".
8. Output JSON only. Do not explain.

Schema:
{MOTORPOLICIES_SCHEMA}

User query:
{user_query}
"""
    try:
        resp = _gemini.generate_content(prompt)
        txt = _safe_extract_text(resp).strip().replace("```json", "").replace("```", "")
        meta = json.loads(txt) if txt else {}
    except Exception:
        meta = {}

    q = user_query.lower()

    target = str(meta.get("target", "")).strip().lower()
    if "loss ratio" in q:
        target = "loss_ratio"
    elif target not in NUMERIC_COLS:
        if "claim" in q:
            target = "claim_amount"
        else:
            target = "gross_premium"

    preds_in = meta.get("predictors", [])
    predictors = []
    if isinstance(preds_in, list):
        for p in preds_in:
            p = str(p).strip().lower()
            if p in NUMERIC_COLS and p != target and p not in predictors:
                predictors.append(p)

    filters = {}
    fin = meta.get("filters", {})
    if isinstance(fin, dict):
        for k, v in fin.items():
            col = str(k).strip().lower()
            if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
                if isinstance(v, (str, int, float)):
                    filters[col] = v

    date_col = str(meta.get("date_col", "")).strip().lower()
    if date_col not in DATE_COLS_ALLOWED:
        if "claim date" in q or "claim_intimation_date" in q:
            date_col = "claim_intimation_date"
        elif "expiry" in q or "end date" in q or "policy_end_date" in q:
            date_col = "policy_end_date"
        else:
            date_col = DEFAULT_DATE_COL

    freq = str(meta.get("freq", "D")).upper()
    if freq not in {"D", "M", "Y"}:
        if re.search(r"\b(month|monthly|months)\b", q):
            freq = "M"
        elif re.search(r"\b(year|yearly|annual)\b", q):
            freq = "Y"
        else:
            freq = "D"

    horizon_days = meta.get("horizon_days", None)
    if not isinstance(horizon_days, int) or horizon_days <= 0:
        horizon_days = _infer_days_from_text(q)
        if horizon_days is None:
            horizon_days = 120 if freq == "M" else (365 if freq == "Y" else 90)

    min_by_freq = {"D": 7, "M": 30, "Y": 365}
    horizon_days = max(min_by_freq.get(freq, 7), int(horizon_days))

    return {
        "target": target,
        "predictors": predictors,
        "filters": filters,
        "horizon_days": horizon_days,
        "freq": freq,
        "date_col": date_col
    }

def _infer_days_from_text(q: str) -> int | None:
    m = re.search(r"next\s+(\d+)\s*days?", q)
    if m: return int(m.group(1))
    m = re.search(r"next\s+(\d+)\s*weeks?", q)
    if m: return int(m.group(1)) * 7
    m = re.search(r"next\s+(\d+)\s*months?", q)
    if m: return int(m.group(1)) * 30
    m = re.search(r"next\s+(\d+)\s*years?", q)
    if m: return int(m.group(1)) * 365
    if re.search(r"next\s*month", q): return 30
    if re.search(r"next\s*year", q): return 365
    return None

def _like(val: str) -> str:
    return "%" + val.replace("%", "\\%").replace("_", "\\_") + "%"

def _boolish_where(col: str, val_raw: str, negate: bool = False) -> str:
    v = str(val_raw).strip().lower()
    neg_vals = "','".join(sorted(NEG_SYNONYMS))
    pos_vals = "','".join(sorted(POS_SYNONYMS))
    if negate:
        return f"(LOWER({col}) NOT IN ('{pos_vals}'))"
    if v in NEG_SYNONYMS:
        return f"(LOWER({col}) IN ('{neg_vals}') OR {col} IS NULL)"
    if v in POS_SYNONYMS:
        return f"(LOWER({col}) IN ('{pos_vals}'))"
    safe_val = val_raw.replace("'", "''")
    return f"(LOWER({col}) = LOWER('{safe_val}') OR LOWER({col}) LIKE LOWER('%{safe_val}%'))"

# ========= SQL Builder ========= #
def build_sql(meta: Dict[str, Any], table: str, user_query: str) -> str:
    date_col = meta["date_col"]
    target = meta["target"]
    predictors = meta.get("predictors", [])
    filters = meta.get("filters", {})

    # 🔹 Special handling for loss_ratio
    if target == "loss_ratio":
        sql = f"""
        SELECT {date_col} AS ts_date,
               SUM(claim_amount) / NULLIF(SUM(gross_premium),0) AS loss_ratio
        FROM {table}
        """
        where_clauses = []
        for col, val in filters.items():
            c = re.sub(r"[^\w]", "", str(col).lower())
            if isinstance(val, (int, float)):
                where_clauses.append(f"{c} = {val}")
                continue
            v = str(val).strip()
            if c in {"manufacturer","model","zone","state","city_tier","fuel_type","business_type"}:
                where_clauses.append(f"LOWER({c}) LIKE LOWER('{_like(v)}')")
            elif c in {"zero_dep_flag","fraud_flag","claim_made_flag"}:
                negate = False
                uq = user_query.lower()
                if c == "zero_dep_flag" and ("non-zero" in uq or "non zero" in uq or "without zero dep" in uq):
                    negate = True
                if c == "fraud_flag" and ("exclude" in uq or "excluding" in uq or "without fraud" in uq):
                    negate = True
                where_clauses.append(_boolish_where(c, v, negate=negate))
            else:
                safe_val = v.replace("'", "''")
                where_clauses.append(f"(LOWER({c}) = LOWER('{safe_val}') OR LOWER({c}) LIKE LOWER('%{safe_val}%'))")
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        sql += f" GROUP BY {date_col} ORDER BY ts_date ASC"
        return sql

    cols = [f"{date_col} AS ts_date", target]
    if predictors:
        cols.extend(predictors)
    cols = list(dict.fromkeys(cols))

    where_clauses = []
    for col, val in filters.items():
        c = re.sub(r"[^\w]", "", str(col).lower())
        if isinstance(val, (int, float)):
            where_clauses.append(f"{c} = {val}")
            continue
        v = str(val).strip()
        if c in {"manufacturer","model","zone","state","city_tier","fuel_type","business_type"}:
            where_clauses.append(f"LOWER({c}) LIKE LOWER('{_like(v)}')")
        elif c in {"zero_dep_flag","fraud_flag","claim_made_flag"}:
            negate = False
            uq = user_query.lower()
            if c == "zero_dep_flag" and ("non-zero" in uq or "non zero" in uq or "without zero dep" in uq):
                negate = True
            if c == "fraud_flag" and ("exclude" in uq or "excluding" in uq or "without fraud" in uq):
                negate = True
            where_clauses.append(_boolish_where(c, v, negate=negate))
        else:
            safe_val = v.replace("'", "''")
            where_clauses.append(f"(LOWER({c}) = LOWER('{safe_val}') OR LOWER({c}) LIKE LOWER('%{safe_val}%'))")

    sql = f"SELECT {', '.join(cols)} FROM {table}"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    sql += " ORDER BY ts_date ASC"
    return sql

# ----------------- Forecast Engine ----------------- #
def pandas_rule(freq: str) -> str:
    return {"D": "D", "M": "MS", "Y": "YS"}.get(freq, "D")

def target_aggregator(target: str) -> str:
    if "amount" in target:
        return "sum"
    if "premium" in target:
        return "mean"
    if "sum_insured" in target:
        return "mean"
    if "loss_ratio" in target:
        return "mean"
    return "mean"

def horizon_steps(horizon_days: int, freq: str) -> int:
    if freq == "D":
        return max(1, int(horizon_days))
    if freq == "M":
        return max(1, int(math.ceil(horizon_days / 30)))
    if freq == "Y":
        return max(1, int(math.ceil(horizon_days / 365)))
    return max(1, int(horizon_days))

def aggregate_and_engineer(df_raw: pd.DataFrame, target: str, predictors: List[str], freq: str) -> pd.DataFrame:
    df = df_raw.copy()
    df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    for p in predictors:
        df[p] = pd.to_numeric(df[p], errors="coerce")
    df = df.dropna(subset=["ts_date", target]).sort_values("ts_date")

    rule = pandas_rule(freq)
    agg_fun = target_aggregator(target)

    if agg_fun == "sum":
        y_series = df.set_index("ts_date")[target].resample(rule).sum(min_count=1)
    else:
        y_series = df.set_index("ts_date")[target].resample(rule).mean()

    X_df = pd.DataFrame({"y": y_series})
    for p in predictors:
        X_df[f"x_{p}_mean"] = df.set_index("ts_date")[p].resample(rule).mean()

    X_df["n_obs"] = df.set_index("ts_date")[target].resample(rule).count()
    X_df = X_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    X_df.index.name = "ts"
    X_df.reset_index(inplace=True)

    X_df["dow"] = X_df["ts"].dt.weekday
    X_df["month"] = X_df["ts"].dt.month
    X_df["dow_sin"] = np.sin(2 * np.pi * X_df["dow"] / 7)
    X_df["dow_cos"] = np.cos(2 * np.pi * X_df["dow"] / 7)
    X_df["mon_sin"] = np.sin(2 * np.pi * X_df["month"] / 12)
    X_df["mon_cos"] = np.cos(2 * np.pi * X_df["month"] / 12)

    if freq == "D":
        lags, rolls = [1, 2, 7, 14, 28], [7, 14, 28]
    elif freq == "M":
        lags, rolls = [1, 2, 3, 6, 12], [3, 6, 12]
    else:
        lags, rolls = [1, 2, 3], [2]

    for L in lags:
        X_df[f"y_lag_{L}"] = X_df["y"].shift(L)
    for w in rolls:
        X_df[f"y_roll_mean_{w}"] = X_df["y"].rolling(w).mean().shift(1)
        X_df[f"y_roll_std_{w}"] = X_df["y"].rolling(w).std().shift(1)

    X_df = X_df.dropna()
    if X_df.empty:
        raise ValueError("No valid rows after feature engineering.")
    return X_df

def split_train_test(X_df: pd.DataFrame, horizon_steps_: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X_df) < (horizon_steps_ + 30):
        raise ValueError(f"Not enough history to train and hold out. Need at least {horizon_steps_ + 30}, have {len(X_df)}.")
    X_df = X_df.sort_values("ts").set_index("ts")
    y = X_df["y"].astype(float)
    X = X_df.drop(columns=["y"])
    X_tr, X_te = X.iloc[: -horizon_steps_], X.iloc[-horizon_steps_:]
    y_tr, y_te = y.iloc[: -horizon_steps_], y.iloc[-horizon_steps_:]
    return X_tr, X_te, y_tr, y_te

def train_model(X_tr, y_tr) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
        tree_method="hist",
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model

def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denom = np.where(y_true == 0, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    smape = float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)))))
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2), "sMAPE%": round(smape, 2)}

def recursive_forecast(X_last: pd.DataFrame, model: XGBRegressor, steps: int, freq: str) -> pd.DataFrame:
    idx = X_last.index
    base_t = idx[-1]
    rule = pandas_rule(freq)
    future_idx = pd.date_range(base_t, periods=steps + 1, freq=rule)[1:]

    cur = X_last.iloc[[-1]].copy()
    rows = []
    lag_cols = sorted([c for c in cur.columns if c.startswith("y_lag_")], key=lambda s: int(s.split("_")[-1]))

    for t in future_idx:
        y_hat = float(model.predict(cur)[0])
        rows.append({"ts": t, "y_hat": y_hat})

        for i, c in enumerate(lag_cols):
            if i == 0:
                cur[c] = y_hat
            else:
                cur[c] = cur[lag_cols[i - 1]]

        cur = cur.copy()
        cur.index = pd.DatetimeIndex([t])
        cur["dow"] = t.weekday()
        cur["month"] = t.month
        cur["dow_sin"] = np.sin(2 * np.pi * cur["dow"] / 7)
        cur["dow_cos"] = np.cos(2 * np.pi * cur["dow"] / 7)
        cur["mon_sin"] = np.sin(2 * np.pi * cur["month"] / 12)
        cur["mon_cos"] = np.cos(2 * np.pi * cur["month"] / 12)

    return pd.DataFrame(rows)

# ----------------- Runner ----------------- #
def run_forecast(user_query: str, history: List[Dict[str, str]] | None = None, table: str = "motorpolicies") -> Dict[str, Any]:
    try:
        meta = infer_forecast_metadata(user_query, history)
        target = meta["target"]
        predictors = meta["predictors"]
        horizon_days = int(meta["horizon_days"])
        freq = meta["freq"]
        date_col = meta["date_col"]

        sql = build_sql(meta, table, user_query)
        df = pd.read_sql(text(sql), ENGINE)

        if df.empty:
            return {"error": "No data retrieved for the selected filters.", "sql": sql, "ModuleUsed": "Forecast"}

        # 🔹 recompute loss_ratio if needed
        if target == "loss_ratio":
            if "claim_amount" in df.columns and "gross_premium" in df.columns:
                df["loss_ratio"] = df["claim_amount"].astype(float) / df["gross_premium"].astype(float)
            df = df.dropna(subset=["loss_ratio"])

        df.to_csv("extracted_data.csv", index=False)
        X_df = aggregate_and_engineer(df, target, predictors, freq)
        if len(X_df) < 40:
            return {"error": f"Insufficient history after aggregation: {len(X_df)} points.", "sql": sql, "ModuleUsed": "Forecast"}

        steps = horizon_steps(horizon_days, freq)
        X_tr, X_te, y_tr, y_te = split_train_test(X_df, steps)
        model = train_model(X_tr, y_tr)

        y_back = model.predict(X_te)
        metrics = metrics_report(y_te.to_numpy(), y_back)

        X_full = X_df.sort_values("ts").set_index("ts").drop(columns=["y"])
        fc = recursive_forecast(X_full, model, steps, freq)
        fc.to_csv("forecast_output.csv", index=False)

        safe_fc = fc.astype(str).head(5)
        pipe_table = _df_to_pipe_table(safe_fc, max_rows=5)

        final_text = (
            f"**Forecast Target:** `{target}`\n\n"
            f"**Horizon:** {horizon_days} days ({steps} steps, freq={freq})\n\n"
            f"**Metrics:** {metrics}\n\n"
            f"**Forecast (top 5 rows):**\n{pipe_table}\n\n"
            f"[Download full forecast CSV](/download/forecast_output.csv)"
        )

        return {
            "sql": sql,
            "metrics": metrics,
            "forecast_head": fc.head(5).to_dict(orient="records"),
            "forecast_csv": "forecast_output.csv",
            "extracted_csv": "extracted_data.csv",
            "used_target": target,
            "used_date_col": date_col,
            "used_freq": freq,
            "used_horizon_days": horizon_days,
            "final_text": final_text,
            "table": {
                "columns": safe_fc.columns.tolist(),
                "rows": safe_fc.values.tolist()
            },
            "ModuleUsed": "Forecast"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": f"Forecast failed: {e}", "ModuleUsed": "Forecast"}
    
if __name__ == "__main__":
    chat_history = []
    while True:
        q = input("Forecast request (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        res = run_forecast(q, history=chat_history)
        chat_history.append({"sender": "User", "message": q})
        chat_history.append({"sender": "Bot", "message": str(res)})

        if "error" in res:
            print("\n=== ERROR ===")
            print(res["error"])
            if res.get("sql"):
                print("\nSQL used:")
                print(res["sql"])
            continue

        print("\n=== SQL ===")
        print(res.get("sql"))
        print("\n=== Metrics ===")
        print(res.get("metrics"))
        print("\n=== Forecast (top 5) ===")
        print(pd.DataFrame(res["forecast_head"]))
        print("\nDownload CSV:", res.get("forecast_csv"))
        print("Module:", res.get("ModuleUsed"))




##########################################

# # forecast.py


# import os
# import re
# import json
# import math
# from typing import Dict, Any, List, Tuple

# import numpy as np
# import pandas as pd
# from sqlalchemy import create_engine, text
# from urllib.parse import quote_plus
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from xgboost import XGBRegressor
# import google.generativeai as genai

# from config import (
#     MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
#     GEMINI_API_KEY, GEMINI_MODEL
# )

# # ========= DB Engine ========= #
# def get_engine():
#     url = (
#         f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
#         f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
#     )
#     return create_engine(url, pool_pre_ping=True)

# ENGINE = get_engine()

# # ========= Gemini Setup ========= #
# genai.configure(api_key=GEMINI_API_KEY)
# _gemini = genai.GenerativeModel(GEMINI_MODEL)

# def _safe_extract_text(resp) -> str:
#     try:
#         if hasattr(resp, "text") and resp.text:
#             return resp.text.strip()
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = []
#             for c in resp.candidates:
#                 if hasattr(c, "content") and c.content:
#                     for p in getattr(c.content, "parts", []):
#                         if hasattr(p, "text") and p.text:
#                             parts.append(p.text.strip())
#             if parts:
#                 return "\n".join(parts)
#     except Exception:
#         pass
#     return ""

# # ========= Schema ========= #
# NUMERIC_COLS = {
#     "gross_premium",
#     "claim_amount",
#     "approved_amount",
#     "sum_insured_amount",
#     "vehicle_age",
#     "cc",
#     "days_to_claim_policy_start",
#     "days_before_policy_expiry",
#     "claim_amt_idv_ratio"
# }

# CATEGORICAL_COLS = {
#     "policy_number","manufacturer","model","zone","state","city_tier",
#     "fuel_type","zero_dep_flag","claim_made_flag","business_type","fraud_flag"
# }

# DATE_COLS_ALLOWED = {"policy_start_date","policy_end_date","claim_intimation_date"}
# DEFAULT_DATE_COL = "policy_start_date"

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

# NEG_SYNONYMS = {"no","n","false","0","non","non-zero dep","non zero dep","nonzerodep","without","exclude","excluding"}
# POS_SYNONYMS = {"yes","y","true","1","zero dep","zerodep","fraud","with","include","including"}

# # ========= Metadata Inference ========= #
# def infer_forecast_metadata(user_query: str) -> Dict[str, Any]:
#     prompt = f"""
# You are a Data Science forecasting assistant.

# Return strictly valid JSON with keys:
# - target
# - predictors
# - filters
# - horizon_days
# - freq
# - date_col

# Rules:
# 1. Do NOT invent or assume columns unless the user explicitly mentions them.
# 2. If user does not mention a numeric predictor, leave "predictors": [].
# 3. If user does not mention filters, leave "filters": {{}}.
# 4. Always keep "target" limited to one numeric column (claim_amount, gross_premium, approved_amount, sum_insured_amount, etc.).
# 5. Default "date_col" = "policy_start_date" unless the user explicitly names another.
# 6. Convert months/years to days for "horizon_days".
# 7. "freq" must be one of "D", "M", "Y". If not clear, default = "D".
# 8. Output JSON only. Do not explain.

# Schema:
# {MOTORPOLICIES_SCHEMA}

# User query:
# {user_query}
# """
#     try:
#         resp = _gemini.generate_content(prompt)
#         txt = _safe_extract_text(resp).strip().replace("```json", "").replace("```", "")
#         meta = json.loads(txt) if txt else {}
#     except Exception:
#         meta = {}

#     # --- Post-process & enforce rules ---
#     target = str(meta.get("target", "")).strip().lower()
#     if target not in NUMERIC_COLS:
#         if re.search(r"\b(claim)\b", user_query, flags=re.I):
#             target = "claim_amount"
#         else:
#             target = "gross_premium"

#     preds_in = meta.get("predictors", [])
#     predictors = []
#     if isinstance(preds_in, list):
#         for p in preds_in:
#             p = str(p).strip().lower()
#             if p in NUMERIC_COLS and p != target and p not in predictors:
#                 predictors.append(p)

#     filters = {}
#     fin = meta.get("filters", {})
#     if isinstance(fin, dict):
#         for k, v in fin.items():
#             col = str(k).strip().lower()
#             if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
#                 if isinstance(v, (str, int, float)):
#                     filters[col] = v

#     date_col = str(meta.get("date_col", "")).strip().lower()
#     if date_col not in DATE_COLS_ALLOWED:
#         q = user_query.lower()
#         if "claim date" in q or "claim_intimation_date" in q:
#             date_col = "claim_intimation_date"
#         elif "expiry" in q or "end date" in q or "policy_end_date" in q:
#             date_col = "policy_end_date"
#         else:
#             date_col = DEFAULT_DATE_COL

#     freq = str(meta.get("freq", "D")).upper()
#     if freq not in {"D", "M", "Y"}:
#         q = user_query.lower()
#         if re.search(r"\b(month|monthly|months)\b", q):
#             freq = "M"
#         elif re.search(r"\b(year|yearly|annual)\b", q):
#             freq = "Y"
#         else:
#             freq = "D"

#     horizon_days = meta.get("horizon_days", None)
#     if not isinstance(horizon_days, int) or horizon_days <= 0:
#         q = user_query.lower()
#         horizon_days = _infer_days_from_text(q)
#         if horizon_days is None:
#             horizon_days = 120 if freq == "M" else (365 if freq == "Y" else 90)

#     min_by_freq = {"D": 7, "M": 30, "Y": 365}
#     horizon_days = max(min_by_freq.get(freq, 7), int(horizon_days))

#     return {
#         "target": target,
#         "predictors": predictors,
#         "filters": filters,
#         "horizon_days": horizon_days,
#         "freq": freq,
#         "date_col": date_col
#     }

# def _infer_days_from_text(q: str) -> int | None:
#     m = re.search(r"next\s+(\d+)\s*days?", q)
#     if m: return int(m.group(1))
#     m = re.search(r"next\s+(\d+)\s*weeks?", q)
#     if m: return int(m.group(1)) * 7
#     m = re.search(r"next\s+(\d+)\s*months?", q)
#     if m: return int(m.group(1)) * 30
#     m = re.search(r"next\s+(\d+)\s*years?", q)
#     if m: return int(m.group(1)) * 365
#     if re.search(r"next\s*month", q): return 30
#     if re.search(r"next\s*year", q): return 365
#     return None

# def _like(val: str) -> str:
#     return "%" + val.replace("%", "\\%").replace("_", "\\_") + "%"

# def _boolish_where(col: str, val_raw: str, negate: bool = False) -> str:
#     v = str(val_raw).strip().lower()
#     neg_vals = "','".join(sorted(NEG_SYNONYMS))
#     pos_vals = "','".join(sorted(POS_SYNONYMS))
#     if negate:
#         return f"(LOWER({col}) NOT IN ('{pos_vals}'))"
#     if v in NEG_SYNONYMS:
#         return f"(LOWER({col}) IN ('{neg_vals}') OR {col} IS NULL)"
#     if v in POS_SYNONYMS:
#         return f"(LOWER({col}) IN ('{pos_vals}'))"

#     safe_val = val_raw.replace("'", "''")
#     return f"(LOWER({col}) = LOWER('{safe_val}') OR LOWER({col}) LIKE LOWER('%{safe_val}%'))"

# # ========= SQL Builder ========= #
# def build_sql(meta: Dict[str, Any], table: str, user_query: str) -> str:
#     date_col = meta["date_col"]
#     target = meta["target"]
#     predictors = meta.get("predictors", [])
#     filters = meta.get("filters", {})

#     # --- Select columns ---
#     cols = [f"{date_col} AS ts_date", target]
#     if predictors:  # only add if not empty
#         cols.extend(predictors)
#     cols = list(dict.fromkeys(cols))  # deduplicate

#     # --- Build WHERE ---
#     where_clauses = []
#     for col, val in filters.items():
#         c = re.sub(r"[^\w]", "", str(col).lower())
#         if isinstance(val, (int, float)):
#             where_clauses.append(f"{c} = {val}")
#             continue

#         v = str(val).strip()
#         if c in {"manufacturer","model","zone","state","city_tier","fuel_type","business_type"}:
#             where_clauses.append(f"LOWER({c}) LIKE LOWER('{_like(v)}')")
#         elif c in {"zero_dep_flag","fraud_flag","claim_made_flag"}:
#             negate = False
#             uq = user_query.lower()
#             if c == "zero_dep_flag" and ("non-zero" in uq or "non zero" in uq or "without zero dep" in uq):
#                 negate = True
#             if c == "fraud_flag" and ("exclude" in uq or "excluding" in uq or "without fraud" in uq):
#                 negate = True
#             where_clauses.append(_boolish_where(c, v, negate=negate))
#         else:
#             safe_val = v.replace("'", "''")
#             where_clauses.append(f"(LOWER({c}) = LOWER('{safe_val}') OR LOWER({c}) LIKE LOWER('%{safe_val}%'))")

#     # --- Assemble final SQL ---
#     sql = f"SELECT {', '.join(cols)} FROM {table}"
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " ORDER BY ts_date ASC"
#     return sql

# # ----------------- Forecast Engine ----------------- #
# def pandas_rule(freq: str) -> str:
#     return {"D": "D", "M": "MS", "Y": "YS"}.get(freq, "D")

# def target_aggregator(target: str) -> str:
#     if "amount" in target:
#         return "sum"
#     if "premium" in target:
#         return "mean"
#     if "sum_insured" in target:
#         return "mean"
#     return "mean"

# def horizon_steps(horizon_days: int, freq: str) -> int:
#     if freq == "D":
#         return max(1, int(horizon_days))
#     if freq == "M":
#         return max(1, int(math.ceil(horizon_days / 30)))
#     if freq == "Y":
#         return max(1, int(math.ceil(horizon_days / 365)))
#     return max(1, int(horizon_days))

# def aggregate_and_engineer(df_raw: pd.DataFrame, target: str, predictors: List[str], freq: str) -> pd.DataFrame:
#     df = df_raw.copy()
#     df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
#     df[target] = pd.to_numeric(df[target], errors="coerce")
#     for p in predictors:
#         df[p] = pd.to_numeric(df[p], errors="coerce")
#     df = df.dropna(subset=["ts_date", target]).sort_values("ts_date")

#     rule = pandas_rule(freq)
#     agg_fun = target_aggregator(target)

#     if agg_fun == "sum":
#         y_series = df.set_index("ts_date")[target].resample(rule).sum(min_count=1)
#     else:
#         y_series = df.set_index("ts_date")[target].resample(rule).mean()

#     X_df = pd.DataFrame({"y": y_series})
#     for p in predictors:
#         X_df[f"x_{p}_mean"] = df.set_index("ts_date")[p].resample(rule).mean()

#     X_df["n_obs"] = df.set_index("ts_date")[target].resample(rule).count()
#     X_df = X_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
#     X_df.index.name = "ts"
#     X_df.reset_index(inplace=True)

#     X_df["dow"] = X_df["ts"].dt.weekday
#     X_df["month"] = X_df["ts"].dt.month
#     X_df["dow_sin"] = np.sin(2 * np.pi * X_df["dow"] / 7)
#     X_df["dow_cos"] = np.cos(2 * np.pi * X_df["dow"] / 7)
#     X_df["mon_sin"] = np.sin(2 * np.pi * X_df["month"] / 12)
#     X_df["mon_cos"] = np.cos(2 * np.pi * X_df["month"] / 12)

#     if freq == "D":
#         lags, rolls = [1, 2, 7, 14, 28], [7, 14, 28]
#     elif freq == "M":
#         lags, rolls = [1, 2, 3, 6, 12], [3, 6, 12]
#     else:
#         lags, rolls = [1, 2, 3], [2]

#     for L in lags:
#         X_df[f"y_lag_{L}"] = X_df["y"].shift(L)
#     for w in rolls:
#         X_df[f"y_roll_mean_{w}"] = X_df["y"].rolling(w).mean().shift(1)
#         X_df[f"y_roll_std_{w}"] = X_df["y"].rolling(w).std().shift(1)

#     X_df = X_df.dropna()
#     if X_df.empty:
#         raise ValueError("No valid rows after feature engineering.")
#     return X_df

# def split_train_test(X_df: pd.DataFrame, horizon_steps_: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     if len(X_df) < (horizon_steps_ + 30):
#         raise ValueError(f"Not enough history to train and hold out. Need at least {horizon_steps_ + 30}, have {len(X_df)}.")
#     X_df = X_df.sort_values("ts").set_index("ts")
#     y = X_df["y"].astype(float)
#     X = X_df.drop(columns=["y"])
#     X_tr, X_te = X.iloc[: -horizon_steps_], X.iloc[-horizon_steps_:]
#     y_tr, y_te = y.iloc[: -horizon_steps_], y.iloc[-horizon_steps_:]
#     return X_tr, X_te, y_tr, y_te

# def train_model(X_tr, y_tr) -> XGBRegressor:
#     model = XGBRegressor(
#         n_estimators=400,
#         learning_rate=0.08,
#         max_depth=6,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         random_state=42,
#         n_jobs=-1,
#         objective="reg:squarederror",
#         tree_method="hist",
#         verbosity=0,
#     )
#     model.fit(X_tr, y_tr)
#     return model

# def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     mae = float(mean_absolute_error(y_true, y_pred))
#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     denom = np.where(y_true == 0, 1.0, np.abs(y_true))
#     mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
#     smape = float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)))))
#     return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2), "sMAPE%": round(smape, 2)}

# def recursive_forecast(X_last: pd.DataFrame, model: XGBRegressor, steps: int, freq: str) -> pd.DataFrame:
#     idx = X_last.index
#     base_t = idx[-1]
#     rule = pandas_rule(freq)
#     future_idx = pd.date_range(base_t, periods=steps + 1, freq=rule)[1:]

#     cur = X_last.iloc[[-1]].copy()
#     rows = []
#     lag_cols = sorted([c for c in cur.columns if c.startswith("y_lag_")], key=lambda s: int(s.split("_")[-1]))

#     for t in future_idx:
#         y_hat = float(model.predict(cur)[0])
#         rows.append({"ts": t, "y_hat": y_hat})

#         for i, c in enumerate(lag_cols):
#             if i == 0:
#                 cur[c] = y_hat
#             else:
#                 cur[c] = cur[lag_cols[i - 1]]

#         cur = cur.copy()
#         cur.index = pd.DatetimeIndex([t])
#         cur["dow"] = t.weekday()
#         cur["month"] = t.month
#         cur["dow_sin"] = np.sin(2 * np.pi * cur["dow"] / 7)
#         cur["dow_cos"] = np.cos(2 * np.pi * cur["dow"] / 7)
#         cur["mon_sin"] = np.sin(2 * np.pi * cur["month"] / 12)
#         cur["mon_cos"] = np.cos(2 * np.pi * cur["month"] / 12)

#     return pd.DataFrame(rows)

# # ----------------- Runner ----------------- #
# def run_forecast(user_query: str, table: str = "motorpolicies") -> Dict[str, Any]:
#     try:
#         meta = infer_forecast_metadata(user_query)
#         target = meta["target"]
#         predictors = meta["predictors"]
#         horizon_days = int(meta["horizon_days"])
#         freq = meta["freq"]
#         date_col = meta["date_col"]

#         sql = build_sql(meta, table, user_query)
#         df = pd.read_sql(text(sql), ENGINE)
#         if df.empty:
#             return {"error": "No data retrieved for the selected filters.", "sql": sql, "ModuleUsed": "Forecast"}

#         df.to_csv("extracted_data.csv", index=False)

#         X_df = aggregate_and_engineer(df, target, predictors, freq)
#         if len(X_df) < 40:
#             return {"error": f"Insufficient history after aggregation: {len(X_df)} points.", "sql": sql, "ModuleUsed": "Forecast"}

#         steps = horizon_steps(horizon_days, freq)

#         X_tr, X_te, y_tr, y_te = split_train_test(X_df, steps)
#         model = train_model(X_tr, y_tr)

#         y_back = model.predict(X_te)
#         metrics = metrics_report(y_te.to_numpy(), y_back)

#         X_full = X_df.sort_values("ts").set_index("ts").drop(columns=["y"])
#         fc = recursive_forecast(X_full, model, steps, freq)
#         fc.to_csv("forecast_output.csv", index=False)

#         return {
#             "sql": sql,
#             "metrics": metrics,
#             "forecast_head": fc.head(12).to_dict(orient="records"),
#             "forecast_csv": "forecast_output.csv",
#             "extracted_csv": "extracted_data.csv",
#             "used_target": target,
#             "used_date_col": date_col,
#             "used_freq": freq,
#             "used_horizon_days": horizon_days,
#             "ModuleUsed": "Forecast"
#         }

#     except Exception as e:
#         return {"error": f"Forecast failed: {e}", "ModuleUsed": "Forecast"}

# if __name__ == "__main__":
#     while True:
#         q = input("Forecast request (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         res = run_forecast(q)
#         err = res.get("error")
#         if err:
#             print("\n=== ERROR ===")
#             print(err)
#             if res.get("sql"):
#                 print("\nSQL used:")
#                 print(res["sql"])
#             print("Module:", res.get("ModuleUsed"))
#             continue

#         print("\n=== SQL ===")
#         print(res.get("sql"))
#         print("\n=== Metrics ===")
#         print(res.get("metrics"))
#         print("\n=== Forecast (head) ===")
#         fh = res.get("forecast_head")
#         if isinstance(fh, list):
#             print(pd.DataFrame(fh))
#         else:
#             print(fh)
#         print("\nExtracted:", res.get("extracted_csv"))
#         print("Forecast:", res.get("forecast_csv"))
#         print("Target:", res.get("used_target"), "| Date:", res.get("used_date_col"), "| Freq:", res.get("used_freq"), "| HorizonDays:", res.get("used_horizon_days"))
#         print("Module:", res.get("ModuleUsed"))





#############################


# # forecast.py

# import os
# import re
# import json
# import math
# from typing import Dict, Any, List, Tuple

# import numpy as np
# import pandas as pd
# from sqlalchemy import create_engine, text
# from urllib.parse import quote_plus
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from xgboost import XGBRegressor
# import google.generativeai as genai

# from config import (
#     MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
#     GEMINI_API_KEY, GEMINI_MODEL
# )

# # ========= DB Engine ========= #
# def get_engine():
#     url = (
#         f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
#         f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
#     )
#     return create_engine(url, pool_pre_ping=True)

# ENGINE = get_engine()

# # ========= Gemini Setup ========= #
# genai.configure(api_key=GEMINI_API_KEY)
# _gemini = genai.GenerativeModel(GEMINI_MODEL)

# def _safe_extract_text(resp) -> str:
#     try:
#         if hasattr(resp, "text") and resp.text:
#             return resp.text.strip()
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = []
#             for c in resp.candidates:
#                 if hasattr(c, "content") and c.content:
#                     for p in getattr(c.content, "parts", []):
#                         if hasattr(p, "text") and p.text:
#                             parts.append(p.text.strip())
#             if parts:
#                 return "\n".join(parts)
#     except Exception:
#         pass
#     return ""

# # ========= Schema ========= #
# NUMERIC_COLS = {
#     "gross_premium",
#     "claim_amount",
#     "approved_amount",
#     "sum_insured_amount",
#     "vehicle_age",
#     "cc",
#     "days_to_claim_policy_start",
#     "days_before_policy_expiry",
#     "claim_amt_idv_ratio"
# }

# CATEGORICAL_COLS = {
#     "policy_number","manufacturer","model","zone","state","city_tier",
#     "fuel_type","zero_dep_flag","claim_made_flag","business_type","fraud_flag"
# }

# DATE_COLS_ALLOWED = {"policy_start_date","policy_end_date","claim_intimation_date"}
# DEFAULT_DATE_COL = "policy_start_date"

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

# NEG_SYNONYMS = {"no","n","false","0","non","non-zero dep","non zero dep","nonzerodep","without","exclude","excluding"}
# POS_SYNONYMS = {"yes","y","true","1","zero dep","zerodep","fraud","with","include","including"}

# # ========= Helpers ========= #
# def _format_history(history: List[Dict[str, str]]) -> str:
#     """Format chat history into a readable string for Gemini context."""
#     if not history:
#         return "No prior history."
#     lines = []
#     for h in history[-10:]:  # last 5 turns
#         role = h.get("sender", "User")
#         msg = h.get("message", "").strip()
#         lines.append(f"{role}: {msg}")
#     return "\n".join(lines)

# # ========= Metadata Inference ========= #
# def infer_forecast_metadata(user_query: str, history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
#     history_str = _format_history(history) if history else "No prior history."

#     prompt = f"""
# You are a Data Science forecasting assistant.

# Conversation so far:
# {history_str}

# Now analyze the latest query.

# Return strictly valid JSON with keys:
# - target
# - predictors
# - filters
# - horizon_days
# - freq
# - date_col

# Rules:
# 1. Do NOT invent or assume columns unless the user explicitly mentions them.
# 2. If user does not mention a numeric predictor, leave "predictors": [].
# 3. If user does not mention filters, leave "filters": {{}}.
# 4. Always keep "target" limited to one numeric column (claim_amount, gross_premium, approved_amount, sum_insured_amount, etc.).
# 5. Default "date_col" = "policy_start_date" unless the user explicitly names another.
# 6. Convert months/years to days for "horizon_days".
# 7. "freq" must be one of "D", "M", "Y". If not clear, default = "D".
# 8. Output JSON only. Do not explain.

# Schema:
# {MOTORPOLICIES_SCHEMA}

# User query:
# {user_query}
# """
#     try:
#         resp = _gemini.generate_content(prompt)
#         txt = _safe_extract_text(resp).strip().replace("```json", "").replace("```", "")
#         meta = json.loads(txt) if txt else {}
#     except Exception:
#         meta = {}

#     # --- Post-process & enforce rules ---
#     target = str(meta.get("target", "")).strip().lower()
#     if target not in NUMERIC_COLS:
#         if re.search(r"\b(claim)\b", user_query, flags=re.I):
#             target = "claim_amount"
#         else:
#             target = "gross_premium"

#     preds_in = meta.get("predictors", [])
#     predictors = []
#     if isinstance(preds_in, list):
#         for p in preds_in:
#             p = str(p).strip().lower()
#             if p in NUMERIC_COLS and p != target and p not in predictors:
#                 predictors.append(p)

#     filters = {}
#     fin = meta.get("filters", {})
#     if isinstance(fin, dict):
#         for k, v in fin.items():
#             col = str(k).strip().lower()
#             if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
#                 if isinstance(v, (str, int, float)):
#                     filters[col] = v

#     date_col = str(meta.get("date_col", "")).strip().lower()
#     if date_col not in DATE_COLS_ALLOWED:
#         q = user_query.lower()
#         if "claim date" in q or "claim_intimation_date" in q:
#             date_col = "claim_intimation_date"
#         elif "expiry" in q or "end date" in q or "policy_end_date" in q:
#             date_col = "policy_end_date"
#         else:
#             date_col = DEFAULT_DATE_COL

#     freq = str(meta.get("freq", "D")).upper()
#     if freq not in {"D", "M", "Y"}:
#         q = user_query.lower()
#         if re.search(r"\b(month|monthly|months)\b", q):
#             freq = "M"
#         elif re.search(r"\b(year|yearly|annual)\b", q):
#             freq = "Y"
#         else:
#             freq = "D"

#     horizon_days = meta.get("horizon_days", None)
#     if not isinstance(horizon_days, int) or horizon_days <= 0:
#         q = user_query.lower()
#         horizon_days = _infer_days_from_text(q)
#         if horizon_days is None:
#             horizon_days = 120 if freq == "M" else (365 if freq == "Y" else 90)

#     min_by_freq = {"D": 7, "M": 30, "Y": 365}
#     horizon_days = max(min_by_freq.get(freq, 7), int(horizon_days))

#     return {
#         "target": target,
#         "predictors": predictors,
#         "filters": filters,
#         "horizon_days": horizon_days,
#         "freq": freq,
#         "date_col": date_col
#     }

# def _infer_days_from_text(q: str) -> int | None:
#     m = re.search(r"next\s+(\d+)\s*days?", q)
#     if m: return int(m.group(1))
#     m = re.search(r"next\s+(\d+)\s*weeks?", q)
#     if m: return int(m.group(1)) * 7
#     m = re.search(r"next\s+(\d+)\s*months?", q)
#     if m: return int(m.group(1)) * 30
#     m = re.search(r"next\s+(\d+)\s*years?", q)
#     if m: return int(m.group(1)) * 365
#     if re.search(r"next\s*month", q): return 30
#     if re.search(r"next\s*year", q): return 365
#     return None

# def _like(val: str) -> str:
#     return "%" + val.replace("%", "\\%").replace("_", "\\_") + "%"

# def _boolish_where(col: str, val_raw: str, negate: bool = False) -> str:
#     v = str(val_raw).strip().lower()
#     neg_vals = "','".join(sorted(NEG_SYNONYMS))
#     pos_vals = "','".join(sorted(POS_SYNONYMS))
#     if negate:
#         return f"(LOWER({col}) NOT IN ('{pos_vals}'))"
#     if v in NEG_SYNONYMS:
#         return f"(LOWER({col}) IN ('{neg_vals}') OR {col} IS NULL)"
#     if v in POS_SYNONYMS:
#         return f"(LOWER({col}) IN ('{pos_vals}'))"

#     safe_val = val_raw.replace("'", "''")
#     return f"(LOWER({col}) = LOWER('{safe_val}') OR LOWER({col}) LIKE LOWER('%{safe_val}%'))"

# # ========= SQL Builder ========= #
# def build_sql(meta: Dict[str, Any], table: str, user_query: str) -> str:
#     date_col = meta["date_col"]
#     target = meta["target"]
#     predictors = meta.get("predictors", [])
#     filters = meta.get("filters", {})

#     cols = [f"{date_col} AS ts_date", target]
#     if predictors:
#         cols.extend(predictors)
#     cols = list(dict.fromkeys(cols))

#     where_clauses = []
#     for col, val in filters.items():
#         c = re.sub(r"[^\w]", "", str(col).lower())
#         if isinstance(val, (int, float)):
#             where_clauses.append(f"{c} = {val}")
#             continue

#         v = str(val).strip()
#         if c in {"manufacturer","model","zone","state","city_tier","fuel_type","business_type"}:
#             where_clauses.append(f"LOWER({c}) LIKE LOWER('{_like(v)}')")

#         elif c in {"zero_dep_flag","fraud_flag","claim_made_flag"}:
#             negate = False
#             uq = user_query.lower()
#             if c == "zero_dep_flag" and ("non-zero" in uq or "non zero" in uq or "without zero dep" in uq):
#                 negate = True
#             if c == "fraud_flag" and ("exclude" in uq or "excluding" in uq or "without fraud" in uq):
#                 negate = True
#             where_clauses.append(_boolish_where(c, v, negate=negate))
#         else:
#             safe_val = v.replace("'", "''")
#             where_clauses.append(f"(LOWER({c}) = LOWER('{safe_val}') OR LOWER({c}) LIKE LOWER('%{safe_val}%'))")

#     sql = f"SELECT {', '.join(cols)} FROM {table}"
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " ORDER BY ts_date ASC"
#     return sql

# # ----------------- Forecast Engine ----------------- #
# def pandas_rule(freq: str) -> str:
#     return {"D": "D", "M": "MS", "Y": "YS"}.get(freq, "D")

# def target_aggregator(target: str) -> str:
#     if "amount" in target:
#         return "sum"
#     if "premium" in target:
#         return "mean"
#     if "sum_insured" in target:
#         return "mean"
#     return "mean"

# def horizon_steps(horizon_days: int, freq: str) -> int:
#     if freq == "D":
#         return max(1, int(horizon_days))
#     if freq == "M":
#         return max(1, int(math.ceil(horizon_days / 30)))
#     if freq == "Y":
#         return max(1, int(math.ceil(horizon_days / 365)))
#     return max(1, int(horizon_days))

# def aggregate_and_engineer(df_raw: pd.DataFrame, target: str, predictors: List[str], freq: str) -> pd.DataFrame:
#     df = df_raw.copy()
#     df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
#     df[target] = pd.to_numeric(df[target], errors="coerce")
#     for p in predictors:
#         df[p] = pd.to_numeric(df[p], errors="coerce")
#     df = df.dropna(subset=["ts_date", target]).sort_values("ts_date")

#     rule = pandas_rule(freq)
#     agg_fun = target_aggregator(target)

#     if agg_fun == "sum":
#         y_series = df.set_index("ts_date")[target].resample(rule).sum(min_count=1)
#     else:
#         y_series = df.set_index("ts_date")[target].resample(rule).mean()

#     X_df = pd.DataFrame({"y": y_series})
#     for p in predictors:
#         X_df[f"x_{p}_mean"] = df.set_index("ts_date")[p].resample(rule).mean()

#     X_df["n_obs"] = df.set_index("ts_date")[target].resample(rule).count()
#     X_df = X_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
#     X_df.index.name = "ts"
#     X_df.reset_index(inplace=True)

#     X_df["dow"] = X_df["ts"].dt.weekday
#     X_df["month"] = X_df["ts"].dt.month
#     X_df["dow_sin"] = np.sin(2 * np.pi * X_df["dow"] / 7)
#     X_df["dow_cos"] = np.cos(2 * np.pi * X_df["dow"] / 7)
#     X_df["mon_sin"] = np.sin(2 * np.pi * X_df["month"] / 12)
#     X_df["mon_cos"] = np.cos(2 * np.pi * X_df["month"] / 12)

#     if freq == "D":
#         lags, rolls = [1, 2, 7, 14, 28], [7, 14, 28]
#     elif freq == "M":
#         lags, rolls = [1, 2, 3, 6, 12], [3, 6, 12]
#     else:
#         lags, rolls = [1, 2, 3], [2]

#     for L in lags:
#         X_df[f"y_lag_{L}"] = X_df["y"].shift(L)
#     for w in rolls:
#         X_df[f"y_roll_mean_{w}"] = X_df["y"].rolling(w).mean().shift(1)
#         X_df[f"y_roll_std_{w}"] = X_df["y"].rolling(w).std().shift(1)

#     X_df = X_df.dropna()
#     if X_df.empty:
#         raise ValueError("No valid rows after feature engineering.")
#     return X_df

# def split_train_test(X_df: pd.DataFrame, horizon_steps_: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     if len(X_df) < (horizon_steps_ + 30):
#         raise ValueError(f"Not enough history to train and hold out. Need at least {horizon_steps_ + 30}, have {len(X_df)}.")
#     X_df = X_df.sort_values("ts").set_index("ts")
#     y = X_df["y"].astype(float)
#     X = X_df.drop(columns=["y"])
#     X_tr, X_te = X.iloc[: -horizon_steps_], X.iloc[-horizon_steps_:]
#     y_tr, y_te = y.iloc[: -horizon_steps_], y.iloc[-horizon_steps_:]
#     return X_tr, X_te, y_tr, y_te

# def train_model(X_tr, y_tr) -> XGBRegressor:
#     model = XGBRegressor(
#         n_estimators=400,
#         learning_rate=0.08,
#         max_depth=6,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         random_state=42,
#         n_jobs=-1,
#         objective="reg:squarederror",
#         tree_method="hist",
#         verbosity=0,
#     )
#     model.fit(X_tr, y_tr)
#     return model

# def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     mae = float(mean_absolute_error(y_true, y_pred))
#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     denom = np.where(y_true == 0, 1.0, np.abs(y_true))
#     mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
#     smape = float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)))))
#     return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2), "sMAPE%": round(smape, 2)}

# def recursive_forecast(X_last: pd.DataFrame, model: XGBRegressor, steps: int, freq: str) -> pd.DataFrame:
#     idx = X_last.index
#     base_t = idx[-1]
#     rule = pandas_rule(freq)
#     future_idx = pd.date_range(base_t, periods=steps + 1, freq=rule)[1:]

#     cur = X_last.iloc[[-1]].copy()
#     rows = []
#     lag_cols = sorted([c for c in cur.columns if c.startswith("y_lag_")], key=lambda s: int(s.split("_")[-1]))

#     for t in future_idx:
#         y_hat = float(model.predict(cur)[0])
#         rows.append({"ts": t, "y_hat": y_hat})

#         for i, c in enumerate(lag_cols):
#             if i == 0:
#                 cur[c] = y_hat
#             else:
#                 cur[c] = cur[lag_cols[i - 1]]

#         cur = cur.copy()
#         cur.index = pd.DatetimeIndex([t])
#         cur["dow"] = t.weekday()
#         cur["month"] = t.month
#         cur["dow_sin"] = np.sin(2 * np.pi * cur["dow"] / 7)
#         cur["dow_cos"] = np.cos(2 * np.pi * cur["dow"] / 7)
#         cur["mon_sin"] = np.sin(2 * np.pi * cur["month"] / 12)
#         cur["mon_cos"] = np.cos(2 * np.pi * cur["month"] / 12)

#     return pd.DataFrame(rows)

# # ----------------- Runner ----------------- #
# def run_forecast(user_query: str, history: List[Dict[str, str]] | None = None, table: str = "motorpolicies") -> Dict[str, Any]:
#     try:
#         meta = infer_forecast_metadata(user_query, history)
#         target = meta["target"]
#         predictors = meta["predictors"]
#         horizon_days = int(meta["horizon_days"])
#         freq = meta["freq"]
#         date_col = meta["date_col"]

#         sql = build_sql(meta, table, user_query)
#         df = pd.read_sql(text(sql), ENGINE)
#         if df.empty:
#             return {"error": "No data retrieved for the selected filters.", "sql": sql, "ModuleUsed": "Forecast"}

#         df.to_csv("extracted_data.csv", index=False)

#         X_df = aggregate_and_engineer(df, target, predictors, freq)
#         if len(X_df) < 40:
#             return {"error": f"Insufficient history after aggregation: {len(X_df)} points.", "sql": sql, "ModuleUsed": "Forecast"}

#         steps = horizon_steps(horizon_days, freq)

#         X_tr, X_te, y_tr, y_te = split_train_test(X_df, steps)
#         model = train_model(X_tr, y_tr)

#         y_back = model.predict(X_te)
#         metrics = metrics_report(y_te.to_numpy(), y_back)

#         X_full = X_df.sort_values("ts").set_index("ts").drop(columns=["y"])
#         fc = recursive_forecast(X_full, model, steps, freq)
#         fc.to_csv("forecast_output.csv", index=False)

#         return {
#             "sql": sql,
#             "metrics": metrics,
#             "forecast_head": fc.head(12).to_dict(orient="records"),
#             "forecast_csv": "forecast_output.csv",
#             "extracted_csv": "extracted_data.csv",
#             "used_target": target,
#             "used_date_col": date_col,
#             "used_freq": freq,
#             "used_horizon_days": horizon_days,
#             "ModuleUsed": "Forecast"
#         }

#     except Exception as e:
#         return {"error": f"Forecast failed: {e}", "ModuleUsed": "Forecast"}

# if __name__ == "__main__":
#     chat_history = []
#     while True:
#         q = input("Forecast request (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         res = run_forecast(q, history=chat_history)
#         chat_history.append({"sender": "User", "message": q})
#         chat_history.append({"sender": "Bot", "message": str(res)})
#         err = res.get("error")
#         if err:
#             print("\n=== ERROR ===")
#             print(err)
#             if res.get("sql"):
#                 print("\nSQL used:")
#                 print(res["sql"])
#             print("Module:", res.get("ModuleUsed"))
#             continue

#         print("\n=== SQL ===")
#         print(res.get("sql"))
#         print("\n=== Metrics ===")
#         print(res.get("metrics"))
#         print("\n=== Forecast (head) ===")
#         fh = res.get("forecast_head")
#         if isinstance(fh, list):
#             print(pd.DataFrame(fh))
#         else:
#             print(fh)
#         print("\nExtracted:", res.get("extracted_csv"))
#         print("Forecast:", res.get("forecast_csv"))
#         print("Target:", res.get("used_target"), "| Date:", res.get("used_date_col"), "| Freq:", res.get("used_freq"), "| HorizonDays:", res.get("used_horizon_days"))
#         print("Module:", res.get("ModuleUsed"))
        





##########################################

# # forecast.py


# import os
# import re
# import json
# import math
# from typing import Dict, Any, List, Tuple

# import numpy as np
# import pandas as pd
# from sqlalchemy import create_engine, text
# from urllib.parse import quote_plus
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from xgboost import XGBRegressor
# import google.generativeai as genai

# from config import (
#     MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD,
#     GEMINI_API_KEY, GEMINI_MODEL
# )

# # ========= DB Engine ========= #
# def get_engine():
#     url = (
#         f"mysql+pymysql://{MYSQL_USER}:{quote_plus(MYSQL_PASSWORD)}"
#         f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
#     )
#     return create_engine(url, pool_pre_ping=True)

# ENGINE = get_engine()

# # ========= Gemini Setup ========= #
# genai.configure(api_key=GEMINI_API_KEY)
# _gemini = genai.GenerativeModel(GEMINI_MODEL)

# def _safe_extract_text(resp) -> str:
#     try:
#         if hasattr(resp, "text") and resp.text:
#             return resp.text.strip()
#         if hasattr(resp, "candidates") and resp.candidates:
#             parts = []
#             for c in resp.candidates:
#                 if hasattr(c, "content") and c.content:
#                     for p in getattr(c.content, "parts", []):
#                         if hasattr(p, "text") and p.text:
#                             parts.append(p.text.strip())
#             if parts:
#                 return "\n".join(parts)
#     except Exception:
#         pass
#     return ""

# # ========= Schema ========= #
# NUMERIC_COLS = {
#     "gross_premium",
#     "claim_amount",
#     "approved_amount",
#     "sum_insured_amount",
#     "vehicle_age",
#     "cc",
#     "days_to_claim_policy_start",
#     "days_before_policy_expiry",
#     "claim_amt_idv_ratio"
# }

# CATEGORICAL_COLS = {
#     "policy_number","manufacturer","model","zone","state","city_tier",
#     "fuel_type","zero_dep_flag","claim_made_flag","business_type","fraud_flag"
# }

# DATE_COLS_ALLOWED = {"policy_start_date","policy_end_date","claim_intimation_date"}
# DEFAULT_DATE_COL = "policy_start_date"

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

# NEG_SYNONYMS = {"no","n","false","0","non","non-zero dep","non zero dep","nonzerodep","without","exclude","excluding"}
# POS_SYNONYMS = {"yes","y","true","1","zero dep","zerodep","fraud","with","include","including"}

# # ========= Metadata Inference ========= #
# def infer_forecast_metadata(user_query: str) -> Dict[str, Any]:
#     prompt = f"""
# You are a Data Science forecasting assistant.

# Return strictly valid JSON with keys:
# - target
# - predictors
# - filters
# - horizon_days
# - freq
# - date_col

# Rules:
# 1. Do NOT invent or assume columns unless the user explicitly mentions them.
# 2. If user does not mention a numeric predictor, leave "predictors": [].
# 3. If user does not mention filters, leave "filters": {{}}.
# 4. Always keep "target" limited to one numeric column (claim_amount, gross_premium, approved_amount, sum_insured_amount, etc.).
# 5. Default "date_col" = "policy_start_date" unless the user explicitly names another.
# 6. Convert months/years to days for "horizon_days".
# 7. "freq" must be one of "D", "M", "Y". If not clear, default = "D".
# 8. Output JSON only. Do not explain.

# Schema:
# {MOTORPOLICIES_SCHEMA}

# User query:
# {user_query}
# """
#     try:
#         resp = _gemini.generate_content(prompt)
#         txt = _safe_extract_text(resp).strip().replace("```json", "").replace("```", "")
#         meta = json.loads(txt) if txt else {}
#     except Exception:
#         meta = {}

#     # --- Post-process & enforce rules ---
#     target = str(meta.get("target", "")).strip().lower()
#     if target not in NUMERIC_COLS:
#         if re.search(r"\b(claim)\b", user_query, flags=re.I):
#             target = "claim_amount"
#         else:
#             target = "gross_premium"

#     preds_in = meta.get("predictors", [])
#     predictors = []
#     if isinstance(preds_in, list):
#         for p in preds_in:
#             p = str(p).strip().lower()
#             if p in NUMERIC_COLS and p != target and p not in predictors:
#                 predictors.append(p)

#     filters = {}
#     fin = meta.get("filters", {})
#     if isinstance(fin, dict):
#         for k, v in fin.items():
#             col = str(k).strip().lower()
#             if col in CATEGORICAL_COLS or col in NUMERIC_COLS:
#                 if isinstance(v, (str, int, float)):
#                     filters[col] = v

#     date_col = str(meta.get("date_col", "")).strip().lower()
#     if date_col not in DATE_COLS_ALLOWED:
#         q = user_query.lower()
#         if "claim date" in q or "claim_intimation_date" in q:
#             date_col = "claim_intimation_date"
#         elif "expiry" in q or "end date" in q or "policy_end_date" in q:
#             date_col = "policy_end_date"
#         else:
#             date_col = DEFAULT_DATE_COL

#     freq = str(meta.get("freq", "D")).upper()
#     if freq not in {"D", "M", "Y"}:
#         q = user_query.lower()
#         if re.search(r"\b(month|monthly|months)\b", q):
#             freq = "M"
#         elif re.search(r"\b(year|yearly|annual)\b", q):
#             freq = "Y"
#         else:
#             freq = "D"

#     horizon_days = meta.get("horizon_days", None)
#     if not isinstance(horizon_days, int) or horizon_days <= 0:
#         q = user_query.lower()
#         horizon_days = _infer_days_from_text(q)
#         if horizon_days is None:
#             horizon_days = 120 if freq == "M" else (365 if freq == "Y" else 90)

#     min_by_freq = {"D": 7, "M": 30, "Y": 365}
#     horizon_days = max(min_by_freq.get(freq, 7), int(horizon_days))

#     return {
#         "target": target,
#         "predictors": predictors,
#         "filters": filters,
#         "horizon_days": horizon_days,
#         "freq": freq,
#         "date_col": date_col
#     }

# def _infer_days_from_text(q: str) -> int | None:
#     m = re.search(r"next\s+(\d+)\s*days?", q)
#     if m: return int(m.group(1))
#     m = re.search(r"next\s+(\d+)\s*weeks?", q)
#     if m: return int(m.group(1)) * 7
#     m = re.search(r"next\s+(\d+)\s*months?", q)
#     if m: return int(m.group(1)) * 30
#     m = re.search(r"next\s+(\d+)\s*years?", q)
#     if m: return int(m.group(1)) * 365
#     if re.search(r"next\s*month", q): return 30
#     if re.search(r"next\s*year", q): return 365
#     return None

# def _like(val: str) -> str:
#     return "%" + val.replace("%", "\\%").replace("_", "\\_") + "%"

# def _boolish_where(col: str, val_raw: str, negate: bool = False) -> str:
#     v = str(val_raw).strip().lower()
#     neg_vals = "','".join(sorted(NEG_SYNONYMS))
#     pos_vals = "','".join(sorted(POS_SYNONYMS))
#     if negate:
#         return f"(LOWER({col}) NOT IN ('{pos_vals}'))"
#     if v in NEG_SYNONYMS:
#         return f"(LOWER({col}) IN ('{neg_vals}') OR {col} IS NULL)"
#     if v in POS_SYNONYMS:
#         return f"(LOWER({col}) IN ('{pos_vals}'))"

#     safe_val = val_raw.replace("'", "''")
#     return f"(LOWER({col}) = LOWER('{safe_val}') OR LOWER({col}) LIKE LOWER('%{safe_val}%'))"

# # ========= SQL Builder ========= #
# def build_sql(meta: Dict[str, Any], table: str, user_query: str) -> str:
#     date_col = meta["date_col"]
#     target = meta["target"]
#     predictors = meta.get("predictors", [])
#     filters = meta.get("filters", {})

#     # --- Select columns ---
#     cols = [f"{date_col} AS ts_date", target]
#     if predictors:  # only add if not empty
#         cols.extend(predictors)
#     cols = list(dict.fromkeys(cols))  # deduplicate

#     # --- Build WHERE ---
#     where_clauses = []
#     for col, val in filters.items():
#         c = re.sub(r"[^\w]", "", str(col).lower())
#         if isinstance(val, (int, float)):
#             where_clauses.append(f"{c} = {val}")
#             continue

#         v = str(val).strip()
#         if c in {"manufacturer","model","zone","state","city_tier","fuel_type","business_type"}:
#             where_clauses.append(f"LOWER({c}) LIKE LOWER('{_like(v)}')")
#         elif c in {"zero_dep_flag","fraud_flag","claim_made_flag"}:
#             negate = False
#             uq = user_query.lower()
#             if c == "zero_dep_flag" and ("non-zero" in uq or "non zero" in uq or "without zero dep" in uq):
#                 negate = True
#             if c == "fraud_flag" and ("exclude" in uq or "excluding" in uq or "without fraud" in uq):
#                 negate = True
#             where_clauses.append(_boolish_where(c, v, negate=negate))
#         else:
#             safe_val = v.replace("'", "''")
#             where_clauses.append(f"(LOWER({c}) = LOWER('{safe_val}') OR LOWER({c}) LIKE LOWER('%{safe_val}%'))")

#     # --- Assemble final SQL ---
#     sql = f"SELECT {', '.join(cols)} FROM {table}"
#     if where_clauses:
#         sql += " WHERE " + " AND ".join(where_clauses)
#     sql += " ORDER BY ts_date ASC"
#     return sql

# # ----------------- Forecast Engine ----------------- #
# def pandas_rule(freq: str) -> str:
#     return {"D": "D", "M": "MS", "Y": "YS"}.get(freq, "D")

# def target_aggregator(target: str) -> str:
#     if "amount" in target:
#         return "sum"
#     if "premium" in target:
#         return "mean"
#     if "sum_insured" in target:
#         return "mean"
#     return "mean"

# def horizon_steps(horizon_days: int, freq: str) -> int:
#     if freq == "D":
#         return max(1, int(horizon_days))
#     if freq == "M":
#         return max(1, int(math.ceil(horizon_days / 30)))
#     if freq == "Y":
#         return max(1, int(math.ceil(horizon_days / 365)))
#     return max(1, int(horizon_days))

# def aggregate_and_engineer(df_raw: pd.DataFrame, target: str, predictors: List[str], freq: str) -> pd.DataFrame:
#     df = df_raw.copy()
#     df["ts_date"] = pd.to_datetime(df["ts_date"], errors="coerce")
#     df[target] = pd.to_numeric(df[target], errors="coerce")
#     for p in predictors:
#         df[p] = pd.to_numeric(df[p], errors="coerce")
#     df = df.dropna(subset=["ts_date", target]).sort_values("ts_date")

#     rule = pandas_rule(freq)
#     agg_fun = target_aggregator(target)

#     if agg_fun == "sum":
#         y_series = df.set_index("ts_date")[target].resample(rule).sum(min_count=1)
#     else:
#         y_series = df.set_index("ts_date")[target].resample(rule).mean()

#     X_df = pd.DataFrame({"y": y_series})
#     for p in predictors:
#         X_df[f"x_{p}_mean"] = df.set_index("ts_date")[p].resample(rule).mean()

#     X_df["n_obs"] = df.set_index("ts_date")[target].resample(rule).count()
#     X_df = X_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
#     X_df.index.name = "ts"
#     X_df.reset_index(inplace=True)

#     X_df["dow"] = X_df["ts"].dt.weekday
#     X_df["month"] = X_df["ts"].dt.month
#     X_df["dow_sin"] = np.sin(2 * np.pi * X_df["dow"] / 7)
#     X_df["dow_cos"] = np.cos(2 * np.pi * X_df["dow"] / 7)
#     X_df["mon_sin"] = np.sin(2 * np.pi * X_df["month"] / 12)
#     X_df["mon_cos"] = np.cos(2 * np.pi * X_df["month"] / 12)

#     if freq == "D":
#         lags, rolls = [1, 2, 7, 14, 28], [7, 14, 28]
#     elif freq == "M":
#         lags, rolls = [1, 2, 3, 6, 12], [3, 6, 12]
#     else:
#         lags, rolls = [1, 2, 3], [2]

#     for L in lags:
#         X_df[f"y_lag_{L}"] = X_df["y"].shift(L)
#     for w in rolls:
#         X_df[f"y_roll_mean_{w}"] = X_df["y"].rolling(w).mean().shift(1)
#         X_df[f"y_roll_std_{w}"] = X_df["y"].rolling(w).std().shift(1)

#     X_df = X_df.dropna()
#     if X_df.empty:
#         raise ValueError("No valid rows after feature engineering.")
#     return X_df

# def split_train_test(X_df: pd.DataFrame, horizon_steps_: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     if len(X_df) < (horizon_steps_ + 30):
#         raise ValueError(f"Not enough history to train and hold out. Need at least {horizon_steps_ + 30}, have {len(X_df)}.")
#     X_df = X_df.sort_values("ts").set_index("ts")
#     y = X_df["y"].astype(float)
#     X = X_df.drop(columns=["y"])
#     X_tr, X_te = X.iloc[: -horizon_steps_], X.iloc[-horizon_steps_:]
#     y_tr, y_te = y.iloc[: -horizon_steps_], y.iloc[-horizon_steps_:]
#     return X_tr, X_te, y_tr, y_te

# def train_model(X_tr, y_tr) -> XGBRegressor:
#     model = XGBRegressor(
#         n_estimators=400,
#         learning_rate=0.08,
#         max_depth=6,
#         subsample=0.9,
#         colsample_bytree=0.9,
#         random_state=42,
#         n_jobs=-1,
#         objective="reg:squarederror",
#         tree_method="hist",
#         verbosity=0,
#     )
#     model.fit(X_tr, y_tr)
#     return model

# def metrics_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#     mae = float(mean_absolute_error(y_true, y_pred))
#     rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
#     denom = np.where(y_true == 0, 1.0, np.abs(y_true))
#     mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
#     smape = float(100.0 * np.mean(2 * np.abs(y_true - y_pred) / np.maximum(1e-9, (np.abs(y_true) + np.abs(y_pred)))))
#     return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 2), "sMAPE%": round(smape, 2)}

# def recursive_forecast(X_last: pd.DataFrame, model: XGBRegressor, steps: int, freq: str) -> pd.DataFrame:
#     idx = X_last.index
#     base_t = idx[-1]
#     rule = pandas_rule(freq)
#     future_idx = pd.date_range(base_t, periods=steps + 1, freq=rule)[1:]

#     cur = X_last.iloc[[-1]].copy()
#     rows = []
#     lag_cols = sorted([c for c in cur.columns if c.startswith("y_lag_")], key=lambda s: int(s.split("_")[-1]))

#     for t in future_idx:
#         y_hat = float(model.predict(cur)[0])
#         rows.append({"ts": t, "y_hat": y_hat})

#         for i, c in enumerate(lag_cols):
#             if i == 0:
#                 cur[c] = y_hat
#             else:
#                 cur[c] = cur[lag_cols[i - 1]]

#         cur = cur.copy()
#         cur.index = pd.DatetimeIndex([t])
#         cur["dow"] = t.weekday()
#         cur["month"] = t.month
#         cur["dow_sin"] = np.sin(2 * np.pi * cur["dow"] / 7)
#         cur["dow_cos"] = np.cos(2 * np.pi * cur["dow"] / 7)
#         cur["mon_sin"] = np.sin(2 * np.pi * cur["month"] / 12)
#         cur["mon_cos"] = np.cos(2 * np.pi * cur["month"] / 12)

#     return pd.DataFrame(rows)

# # ----------------- Runner ----------------- #
# def run_forecast(user_query: str, table: str = "motorpolicies") -> Dict[str, Any]:
#     try:
#         meta = infer_forecast_metadata(user_query)
#         target = meta["target"]
#         predictors = meta["predictors"]
#         horizon_days = int(meta["horizon_days"])
#         freq = meta["freq"]
#         date_col = meta["date_col"]

#         sql = build_sql(meta, table, user_query)
#         df = pd.read_sql(text(sql), ENGINE)
#         if df.empty:
#             return {"error": "No data retrieved for the selected filters.", "sql": sql, "ModuleUsed": "Forecast"}

#         df.to_csv("extracted_data.csv", index=False)

#         X_df = aggregate_and_engineer(df, target, predictors, freq)
#         if len(X_df) < 40:
#             return {"error": f"Insufficient history after aggregation: {len(X_df)} points.", "sql": sql, "ModuleUsed": "Forecast"}

#         steps = horizon_steps(horizon_days, freq)

#         X_tr, X_te, y_tr, y_te = split_train_test(X_df, steps)
#         model = train_model(X_tr, y_tr)

#         y_back = model.predict(X_te)
#         metrics = metrics_report(y_te.to_numpy(), y_back)

#         X_full = X_df.sort_values("ts").set_index("ts").drop(columns=["y"])
#         fc = recursive_forecast(X_full, model, steps, freq)
#         fc.to_csv("forecast_output.csv", index=False)

#         return {
#             "sql": sql,
#             "metrics": metrics,
#             "forecast_head": fc.head(12).to_dict(orient="records"),
#             "forecast_csv": "forecast_output.csv",
#             "extracted_csv": "extracted_data.csv",
#             "used_target": target,
#             "used_date_col": date_col,
#             "used_freq": freq,
#             "used_horizon_days": horizon_days,
#             "ModuleUsed": "Forecast"
#         }

#     except Exception as e:
#         return {"error": f"Forecast failed: {e}", "ModuleUsed": "Forecast"}

# if __name__ == "__main__":
#     while True:
#         q = input("Forecast request (or 'exit'): ").strip()
#         if q.lower() in {"exit", "quit"}:
#             break
#         res = run_forecast(q)
#         err = res.get("error")
#         if err:
#             print("\n=== ERROR ===")
#             print(err)
#             if res.get("sql"):
#                 print("\nSQL used:")
#                 print(res["sql"])
#             print("Module:", res.get("ModuleUsed"))
#             continue

#         print("\n=== SQL ===")
#         print(res.get("sql"))
#         print("\n=== Metrics ===")
#         print(res.get("metrics"))
#         print("\n=== Forecast (head) ===")
#         fh = res.get("forecast_head")
#         if isinstance(fh, list):
#             print(pd.DataFrame(fh))
#         else:
#             print(fh)
#         print("\nExtracted:", res.get("extracted_csv"))
#         print("Forecast:", res.get("forecast_csv"))
#         print("Target:", res.get("used_target"), "| Date:", res.get("used_date_col"), "| Freq:", res.get("used_freq"), "| HorizonDays:", res.get("used_horizon_days"))
#         print("Module:", res.get("ModuleUsed"))
