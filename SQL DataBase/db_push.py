import pandas as pd
from sqlalchemy import create_engine

# Load your CSV
df = pd.read_csv("Motor_Insurance_Data(in).csv")

# Convert date columns to datetime
date_cols = ["POLICY_START_DATE", "POLICY_END_DATE", "CLAIM_INTIMATION_DATE"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Create SQLAlchemy connection
engine = create_engine("mysql+pymysql://root:1234567890@localhost/insurancedb")

# Append data into the existing table
df.to_sql("motorpolicies", con=engine, if_exists="append", index=False)