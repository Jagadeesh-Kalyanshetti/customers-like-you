from pathlib import Path

import pandas as pd

# Relative paths, always run from the project root. UPPER_CASE = constants that never change
RAW = Path("data/raw/online_retail_II.xlsx")
PROCESSED = Path("data/processed")
OUT = Path("outputs")

# sheet_name=None loads every sheet into a dictionary data structure
sheets = pd.read_excel(RAW, sheet_name=None)

"""
concat matches columns by NAME, so if the sheets differed it would silently create extra half-empty columns. 
Assert stops the script if they differ
"""

cols_first = sheets["Year 2009-2010"].columns.tolist()
cols_second = sheets["Year 2010-2011"].columns.tolist()
assert cols_first == cols_second, "Sheets have different column names"

"""
I stacked the two years vertically so I can work on the whole data together because the time split (DE3) and the matrix (DE4) need one customer's full history in one place.
ignore_index=True gives clean row numbers 0..1,067,370 
"""

df = pd.concat(list(sheets.values()), ignore_index=True)

# I removed the space from the name so I can use one simple name in every later script
df = df.rename(columns={"Customer ID": "CustomerID"})

"""
DE1 profile: one row per column of df, showing its type and how much is missing.
isna() gives True/False per cell; sum() counts the Trues per column;
mean() gives the share of Trues per column (True=1, False=0), times 100 for a percentage
"""

profile = pd.DataFrame(
    {
        "dtype": df.dtypes,
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    }
)

"""
Dataset-level facts BEFORE cleaning: 
DE2's cleaning log will be compared against these.
Counts and dates only, no customer IDs (R4)
"""

summary = pd.Series(
    {
        "rows": len(df),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),

        # Invoice mixes numbers and text, so astype(str) first; cancellations start with "C"
        "cancellation_lines": df["Invoice"].astype(str).str.startswith("C").sum(),
        "quantity_non_positive": (df["Quantity"] <= 0).sum(),
        "price_non_positive": (df["Price"] <= 0).sum(),

        # nunique() counts distinct values and ignores missing ones
        "distinct_customers": df["CustomerID"].nunique(),
        "distinct_products": df["StockCode"].nunique(),
    },
    name="value",
)

"""
Parquet needs one type per column, but Invoice and StockCode mix numbers and text. 
I made them all text so the save doesn't fail
"""

df["Invoice"] = df["Invoice"].astype(str)
df["StockCode"] = df["StockCode"].astype(str)
df["Description"] = df["Description"].astype(str)

# exist_ok=True means no error if the folder already exists, so the script is safe to rerun
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

"""
Reading the Excel takes minutes, so I cache the stacked table as parquet.
Scripts 02 to 06 load that instead. The two CSVs go to outputs/ (counts only)
"""

profile.to_csv(OUT / "data_profile.csv")
summary.to_csv(OUT / "data_summary.csv")
df.to_parquet(PROCESSED / "raw.parquet")

print(f"Saved {len(df):,} rows to {PROCESSED / 'raw.parquet'}")