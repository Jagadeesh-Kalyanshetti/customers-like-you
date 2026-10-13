from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

PROCESSED = Path("data/processed")

# a product needs at least this many different buyers in train
MIN_CUSTOMERS = 10  

# train holds one row per invoice line
train = pd.read_parquet(PROCESSED / "train.parquet")

# one row per customer-product pair
pair_quantities = train.groupby(["CustomerID", "StockCode"])["Quantity"].sum().reset_index()

"""
each pair is one customer, so counting a product's rows gives its number of buyers (counting in train would count the same customer once per invoice line)
"""
customers_per_product = pair_quantities["StockCode"].value_counts()

# keep only products with enough buyers, counted on train only
kept_products = customers_per_product[customers_per_product >= MIN_CUSTOMERS].index
pair_quantities = pair_quantities[pair_quantities["StockCode"].isin(kept_products)].reset_index(drop=True)

# log1p so one huge order doesn't dominate a cell; 1 + keeps a quantity of 1 above zero
pair_quantities["Value"] = np.log1p(pair_quantities["Quantity"])

# each customer and each product once, in order of first appearance
customer_ids = pair_quantities["CustomerID"].unique()
product_ids = pair_quantities["StockCode"].unique()

# the matrix only knows positions, so each ID gets a row or column number
customer_to_row = {}
for i, c in enumerate(customer_ids):
    customer_to_row[c] = i

product_to_col = {}
for j, p in enumerate(product_ids):
    product_to_col[p] = j

# translate every pair into its row number and column number
rows = pair_quantities["CustomerID"].map(customer_to_row)
cols = pair_quantities["StockCode"].map(product_to_col)

# one filled cell per pair; shape is needed because empty cells are not listed
sparse_customer_product_matrix = sp.csr_matrix(
    (pair_quantities["Value"], (rows, cols)),
    shape=(len(customer_ids), len(product_ids)),
)

# exist_ok=True: no error if the folder already exists
PROCESSED.mkdir(parents=True, exist_ok=True)

# these files hold customer IDs, so they stay in data/processed, never in outputs (R4)
sp.save_npz(PROCESSED / "sparse_customer_product_matrix.npz", sparse_customer_product_matrix)

# position in each list = row or column number, so no separate map file is needed
pd.DataFrame({"CustomerID": customer_ids}).to_parquet(PROCESSED / "customer_index.parquet", index=False)
pd.DataFrame({"StockCode": product_ids}).to_parquet(PROCESSED / "product_index.parquet", index=False)

"""
held-out pairs from 03; the product filter removed some products, so a pair can point to a product with no column
"""
test_pairs = pd.read_parquet(PROCESSED / "test_pairs.parquet")

# a pair can only be scored if its customer has a row and its product has a column
does_customer_have_row = test_pairs["CustomerID"].isin(customer_to_row)
does_product_have_col = test_pairs["StockCode"].isin(product_to_col)

test_pairs = test_pairs[does_customer_have_row & does_product_have_col].reset_index(drop=True)
test_pairs.to_parquet(PROCESSED / "test_pairs_final.parquet", index=False)

n_customers, n_products = sparse_customer_product_matrix.shape
print(f"Saved {n_customers:,} x {n_products:,} matrix and {len(test_pairs):,} held-out pairs to {PROCESSED}")