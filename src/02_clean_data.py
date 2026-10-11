from pathlib import Path

import pandas as pd

# Relative paths, always run from the project root. UPPER_CASE = constants that never change
PROCESSED = Path("data/processed")
OUT = Path("outputs")

# I load the parquet cache from script 01, so I don't have to read the slow Excel file again
df = pd.read_parquet(PROCESSED/"raw.parquet")

# I save the starting row count first, because every rule's count and the final check depend on it
rows_start = len(df)

# Rule 1: cancellations. Invoice mixes numbers and text, so astype(str) first; cancellations start with "C"
is_cancellation = df['Invoice'].astype(str).str.startswith("C")

# ~ flips True/False, so I keep everything that is NOT a cancellation
df = df[~is_cancellation]
rows_after_cancellations = len(df)

# Removed = rows I had before this rule minus rows left, so it is only this rule's count
removed_cancellations = rows_start - rows_after_cancellations

# Rule 2: quantity. A zero or negative quantity is a return or an adjustment, not a purchase
is_quantity_positive = df['Quantity']>0

# This mask is already True for the rows I want to keep, so no ~ this time
df = df[is_quantity_positive]
rows_after_quantity = len(df)

# I subtract from the previous step's count, not from rows_start, so each rule only reports its own removals
removed_quantity = rows_after_cancellations - rows_after_quantity

# Rule 3: price. A zero or negative price is not a real sale
is_price_positive = df['Price']>0
df = df[is_price_positive]
rows_after_price = len(df)
removed_price = rows_after_quantity - rows_after_price

"""
Rule 4: StockCode. A real product code starts with 5 digits (variants like 85123A still match).
POST, M, D, BANK CHARGES, vouchers and other odd codes don't, so one pattern rule removes them all
"""

is_normal_product_code = df['StockCode'].str.match(r"^\d{5}")
df = df[is_normal_product_code]
rows_after_product_code = len(df)
removed_product_code = rows_after_price - rows_after_product_code

"""
Rule 5: CustomerID. Without a customer I can't put the row in a customer x product matrix
notna() is True where the ID is present, so no ~ here
"""

is_customerID = df['CustomerID'].notna()
df = df[is_customerID]
rows_after_customerID = len(df)
removed_customerID = rows_after_product_code - rows_after_customerID

# Guard: .all() collapses the True/False column into one answer, so the script stops if any ID is still missing
assert df['CustomerID'].notna().all(), "There are still rows with no CustomerID"

# CustomerID was float (13085.0) because of the missing values; with none left I can safely make it int
df['CustomerID'] = df['CustomerID'].astype(int)

# The cleaning log: one dictionary per rule, each one becomes one row of the table
log_rows = [
    {
"rule": "start",
"rows_removed": 0, 
"rows_remaining": rows_start,
    },

    {
"rule": "cancellations",
"rows_removed": removed_cancellations,
"rows_remaining": rows_after_cancellations,
    },

    {
"rule": "quantity",
"rows_removed": removed_quantity,
"rows_remaining": rows_after_quantity,
    },

    {
"rule": "price",
"rows_removed": removed_price,
"rows_remaining": rows_after_price, 
    },

    {
"rule": "stockcodes",
"rows_removed": removed_product_code,
"rows_remaining": rows_after_product_code,
    },

    {
"rule": "customerID",
"rows_removed": removed_customerID,
"rows_remaining": rows_after_customerID
    }
]

# A list of dictionaries becomes a table: each dictionary is a row, the keys are the column names
cleaning_log = pd.DataFrame(log_rows)

# I check that the log adds up BEFORE adding the total row, otherwise the total would be counted twice
total_rows_removed = cleaning_log['rows_removed'].sum()
assert (rows_start - total_rows_removed) == len(df), "Cleaning log does not add up"

# The total row is a one-row table with the same column names as the log
total_row = pd.DataFrame([
    {
"rule": "total",
"rows_removed": total_rows_removed,
"rows_remaining": rows_after_customerID
    },
])

# I stack the total row under the log; ignore_index=True renumbers the rows 0..6
cleaning_log = pd.concat([cleaning_log, total_row], ignore_index=True)

# exist_ok=True means no error if the folder already exists, so the script is safe to rerun
OUT.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

cleaning_log.to_csv(OUT / "cleaning_log.csv", index=False)

# After filtering, df has gaps in its row numbers; I renumber them from 0 before saving a tidy file
df = df.reset_index(drop=True)
df.to_parquet(PROCESSED / "clean.parquet")

print(f"Saved {len(df):,} rows to {PROCESSED / 'clean.parquet'}")