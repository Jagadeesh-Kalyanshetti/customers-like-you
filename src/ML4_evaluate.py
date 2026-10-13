import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

PROCESSED = Path("data/processed")
OUT = Path("outputs")
K = 200
SEED = 42
TOP_N = 10

# I load the train-only matrix, so the held-out pairs stay unseen (R2)
matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")

# ---- the nightly job, timed in three parts ----

# part 1: fit SVD and build the product fingerprints (length 1 each)
start = time.perf_counter()
svd = TruncatedSVD(n_components=K, random_state=SEED)
svd.fit(matrix)
product_vectors = normalize(svd.components_.T)
fit_seconds = time.perf_counter() - start

# part 2: the "customers who bought this also bought" box for all 3,687 products
start = time.perf_counter()
product_scores = product_vectors @ product_vectors.T
# a product must not recommend itself
np.fill_diagonal(product_scores, -np.inf)
similar_top = np.argsort(-product_scores, axis=1)[:, :TOP_N]
all_products_seconds = time.perf_counter() - start

# part 3: top 10 new products for all 5,473 customers
start = time.perf_counter()
# one taste profile per customer (5,473 x 200)
customer_profiles = matrix @ product_vectors
customer_scores = customer_profiles @ product_vectors.T
# rule out everything each customer already bought
bought_rows, bought_columns = matrix.nonzero()
customer_scores[bought_rows, bought_columns] = -np.inf
customer_top = np.argsort(-customer_scores, axis=1)[:, :TOP_N]
all_customers_seconds = time.perf_counter() - start

# ---- evaluation: the held-out pairs are loaded only now ----

customer_index = pd.read_parquet(PROCESSED / "customer_index.parquet")
product_index = pd.read_parquet(PROCESSED / "product_index.parquet")
test_pairs = pd.read_parquet(PROCESSED / "test_pairs_final.parquet")

# the position in each index file is the matrix row or column number
customer_row_of = {}
for row_number, customer_id in enumerate(customer_index["CustomerID"]):
    customer_row_of[customer_id] = row_number

product_column_of = {}
for column_number, stock_code in enumerate(product_index["StockCode"]):
    product_column_of[stock_code] = column_number

# DE4 already dropped pairs with no row or column, so no KeyError is expected
test_rows = [customer_row_of[c] for c in test_pairs["CustomerID"]]
test_columns = [product_column_of[p] for p in test_pairs["StockCode"]]

# True where a customer bought a product in the test period that is new to them
held_out = np.zeros(matrix.shape, dtype=bool)
held_out[test_rows, test_columns] = True

test_customers = np.unique(test_rows)

# model: does any of the customer's 10 suggestions sit in their held-out products?
model_recommendations = customer_top[test_customers]
model_hit = held_out[test_customers[:, None], model_recommendations].any(axis=1)
model_hit_rate = model_hit.mean()

# baseline (R3): the 10 best sellers (most distinct buyers in train) the customer has not bought
buyers_per_product = matrix.getnnz(axis=0)
# stable sort so ties always come out in the same order (R6)
best_sellers_ranked = np.argsort(-buyers_per_product, kind="stable")

baseline_hits = 0
for row in test_customers:
    bought = set(matrix[row].indices)
    picks = []
    for column in best_sellers_ranked:
        if column not in bought:
            picks.append(column)
        if len(picks) == TOP_N:
            break
    if held_out[row, picks].any():
        baseline_hits += 1
baseline_hit_rate = baseline_hits / len(test_customers)

results = [
    {"measure": "k", "value": K},
    {"measure": "test_customers", "value": len(test_customers)},
    {"measure": "heldout_pairs", "value": len(test_pairs)},
    {"measure": "hit_rate_at_10_svd", "value": round(model_hit_rate, 4)},
    {"measure": "hit_rate_at_10_best_sellers", "value": round(baseline_hit_rate, 4)},
    {"measure": "fit_seconds", "value": round(fit_seconds, 2)},
    {"measure": "all_products_seconds", "value": round(all_products_seconds, 2)},
    {"measure": "all_customers_seconds", "value": round(all_customers_seconds, 2)},
]

# exist_ok means no error if outputs/ exists; a fresh clone has no outputs/
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUT / "recommender_results.csv", index=False)

print("Saved hit rates and timing to outputs/recommender_results.csv")