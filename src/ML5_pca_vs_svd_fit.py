import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import normalize

PROCESSED = Path("data/processed")
OUT = Path("outputs")
SEED = 42
K = 200
TOP_N = 10
BYTES_PER_MB = 1_000_000

# ---- load the train-only matrix ----
matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")
n_customers, n_products = matrix.shape

# ---- fit Truncated SVD, timed, with peak memory ----
tracemalloc.start()
start = time.perf_counter()
svd = TruncatedSVD(n_components=K, random_state=SEED)
svd_customer_factors = svd.fit_transform(matrix)
svd_fit_seconds = time.perf_counter() - start
svd_peak_mb = tracemalloc.get_traced_memory()[1] / BYTES_PER_MB
tracemalloc.stop()

# ---- fit PCA (implicit centring on sparse input), same k, same timing ----
tracemalloc.start()
start = time.perf_counter()
pca = PCA(n_components=K, svd_solver="arpack", random_state=SEED)
pca.fit(matrix)
pca_fit_seconds = time.perf_counter() - start
pca_peak_mb = tracemalloc.get_traced_memory()[1] / BYTES_PER_MB
tracemalloc.stop()

# ---- stored factors: k x (customers + products) x 8 bytes ----
svd_factors_mb = K * (n_customers + n_products) * 8 / BYTES_PER_MB
# PCA also stores the column-average vector (one number per product)
pca_factors_mb = (K * (n_customers + n_products) + n_products) * 8 / BYTES_PER_MB

# ---- SVD: relative reconstruction error, same formula as ML2 ----
matrix_size = np.sqrt((matrix.data ** 2).sum())
rebuilt_size = np.sqrt((svd_customer_factors ** 2).sum())
lost_size = np.sqrt(matrix_size ** 2 - rebuilt_size ** 2)
svd_relative_error = lost_size / matrix_size
pca_variance_explained = pca.explained_variance_ratio_.sum()

# ---- held-out purchases, loaded only now (after both fits) ----
customer_index = pd.read_parquet(PROCESSED / "customer_index.parquet")
product_index = pd.read_parquet(PROCESSED / "product_index.parquet")
test_pairs = pd.read_parquet(PROCESSED / "test_pairs_final.parquet")

customer_row_of = {}
for row_number, customer_id in enumerate(customer_index["CustomerID"]):
    customer_row_of[customer_id] = row_number

product_column_of = {}
for column_number, stock_code in enumerate(product_index["StockCode"]):
    product_column_of[stock_code] = column_number

test_rows = [customer_row_of[c] for c in test_pairs["CustomerID"]]
test_columns = [product_column_of[p] for p in test_pairs["StockCode"]]

held_out = np.zeros(matrix.shape, dtype=bool)
held_out[test_rows, test_columns] = True
test_customers = np.unique(test_rows)


def hit_rate_at_10(product_vectors):
    # same scoring as ML4 for both methods: only the fingerprints differ
    scores = (matrix @ product_vectors) @ product_vectors.T
    scores[matrix.nonzero()] = -np.inf
    top = np.argsort(-scores[test_customers], axis=1)[:, :TOP_N]
    hits = held_out[test_customers[:, None], top].any(axis=1)
    return hits.mean()


svd_hit_rate = hit_rate_at_10(normalize(svd.components_.T))
pca_hit_rate = hit_rate_at_10(normalize(pca.components_.T))

# ML4 saved the baseline; I read it instead of recomputing
ml4 = pd.read_csv(OUT / "recommender_results.csv").set_index("measure")["value"]
baseline_hit_rate = float(ml4["hit_rate_at_10_best_sellers"])
# my fresh SVD number should match ML4's (if not, something changed)
assert abs(svd_hit_rate - float(ml4["hit_rate_at_10_svd"])) < 0.0005

# ---- the ML5 table (brief 4.4) ----
# written from what I measured: component 1 vs total log-quantity per customer (0.968 for both)
COMPONENT_1_PCA = (
    "Mostly overall buying volume: correlation 0.97 with a customer's total "
    "log-quantity; top products are the best sellers, same as SVD"
)
COMPONENT_1_SVD = (
    "Overall buying volume: correlation 0.97 with a customer's total "
    "log-quantity; top products are the best sellers (cake cases, "
    "T-light holder, jumbo bag)"
)

rows = [
    {"measure": "k used", "pca": K, "truncated_svd": K},
    {"measure": "fit_seconds", "pca": round(pca_fit_seconds, 2), "truncated_svd": round(svd_fit_seconds, 2)},
    {"measure": "peak_memory_mb", "pca": round(pca_peak_mb, 1), "truncated_svd": round(svd_peak_mb, 1)},
    {"measure": "stored_factors_mb", "pca": round(pca_factors_mb, 2), "truncated_svd": round(svd_factors_mb, 2)},
    {"measure": "variance_explained", "pca": round(pca_variance_explained, 4), "truncated_svd": "not used"},
    {"measure": "relative_reconstruction_error", "pca": "not used", "truncated_svd": round(svd_relative_error, 4)},
    {"measure": "hit_rate_at_10", "pca": round(pca_hit_rate, 4), "truncated_svd": round(svd_hit_rate, 4)},
    {"measure": "best_seller_hit_rate_at_10", "pca": round(baseline_hit_rate, 4), "truncated_svd": round(baseline_hit_rate, 4)},
    {"measure": "component_1_in_words", "pca": COMPONENT_1_PCA, "truncated_svd": COMPONENT_1_SVD},
]

OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "pca_vs_svd_table.csv", index=False)
print("Saved PCA vs SVD table to outputs/pca_vs_svd_table.csv")