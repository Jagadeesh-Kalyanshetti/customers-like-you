from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")
OUT = Path("outputs")

df = pd.read_parquet(PROCESSED / "clean.parquet")

"""
I compute the cutoff from the data instead of typing a date, so the split stays correct and reproducible if the data changes (R6). The test period is the last 2 months.
"""
latest_date = df["InvoiceDate"].max()
test_set_cutoff_date = latest_date - pd.DateOffset(months=2)

"""
The cutoff includes a time of day (12:50), so "<" and ">=" are exact opposites: every row lands in exactly one side, nothing is lost or counted twice.
"""
is_train = df["InvoiceDate"] < test_set_cutoff_date
is_test_candidates = df["InvoiceDate"] >= test_set_cutoff_date

train = df[is_train]
test_candidates = df[is_test_candidates]

"""
R2 guards: no row from the test period may be in train, and no row may go missing. The message prints only if the check fails, so it describes the problem.
"""
assert train["InvoiceDate"].max() < test_set_cutoff_date, "Train contains rows on or after the cutoff"
assert len(train) + len(test_candidates) == len(df), "Train and test rows do not add up to the full data"

"""
I only need to know "did this customer ever buy this product?", so I drop repeated pairs here. 
Quantity is not needed for this lookup; it stays in train for the DE4 matrix.
"""
train_pairs = train[["CustomerID", "StockCode"]].drop_duplicates()

test_pairs = test_candidates[["CustomerID", "StockCode"]].drop_duplicates()
pairs_all = len(test_pairs)

"""
A left merge labels each test pair: "both" = the customer already bought it in train (a repeat), "left_only" = a new product for this customer. Nothing is deleted yet.
"""
merged = test_pairs.merge(train_pairs, on=["CustomerID", "StockCode"], how="left", indicator=True)

"""
I keep only the new products: the box should suggest things the customer does not own yet, and counting repeats would make the model look better than it is (R3).
"""
is_new = merged["_merge"] == "left_only"
new_test_pairs = merged[is_new].drop(columns="_merge")
pairs_after_removing_repeats = len(new_test_pairs)

"""
Customers with no training history have no history to recommend from, so I drop their test pairs, so the model cannot serve them (in real life they would see best sellers). 
"""
train_customers = train["CustomerID"].unique()
is_test_customer_in_train = new_test_pairs["CustomerID"].isin(train_customers)
new_test_pairs = new_test_pairs[is_test_customer_in_train]
pairs_after_removing_new_customers = len(new_test_pairs)

"""
Same reason for products: one that never appears in train has no column, so it can never be recommended. I filter test pairs only; train is never touched.
"""
train_products = train["StockCode"].unique()
is_test_product_in_train = new_test_pairs["StockCode"].isin(train_products)
new_test_pairs = new_test_pairs[is_test_product_in_train]
pairs_after_removing_new_products = len(new_test_pairs)

# exist_ok=True means no error if the folders already exist, so the script is safe to rerun
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# counts only, no customer IDs, so this file is safe to commit (R4)
summary_counts = {
    "cutoff": str(test_set_cutoff_date),
    "train_rows": len(train),
    "test_candidate_rows": len(test_candidates),
    "test_pairs_all": pairs_all,
    "after_removing_repeats": pairs_after_removing_repeats,
    "after_removing_new_customers": pairs_after_removing_new_customers,
    "after_removing_new_products": pairs_after_removing_new_products,
    "test_customers": new_test_pairs["CustomerID"].nunique(),
}

# .items() gives (label, number) pairs, so each pair becomes one row of the table
split_summary = pd.DataFrame(summary_counts.items(), columns=["item", "value"])
split_summary.to_csv(OUT / "split_summary.csv", index=False)

# These files contain customer IDs, so they stay in data/ and are never committed (R4)
train.to_parquet(PROCESSED / "train.parquet", index=False)

# I renumber the held-out pairs from 0 because the filters left gaps in the row labels
new_test_pairs = new_test_pairs.reset_index(drop=True)
new_test_pairs.to_parquet(PROCESSED / "test_pairs.parquet", index=False)

print(f"Saved {len(train):,} train rows and {len(new_test_pairs):,} held-out pairs to {PROCESSED}")