from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

PROCESSED = Path("data/processed")
K = 200
SEED = 42
TOP_N = 10

# I load the train-only matrix, so the held-out pairs stay unseen (R2)
matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")
product_index = pd.read_parquet(PROCESSED / "product_index.parquet")

# k = 200 was fixed in ML2 before any held-out scoring
svd = TruncatedSVD(n_components=K, random_state=SEED)
svd.fit(matrix)

# one row per product (3,687 x 200), each scaled to length 1,
# so a plain multiplication of two rows is their cosine similarity
product_vectors = normalize(svd.components_.T)


def similar_products(product_number, n=TOP_N):
    # multiplying by one product's fingerprint scores every product against it
    scores = product_vectors @ product_vectors[product_number]
    # a product is always most similar to itself, so I rule it out
    scores[product_number] = -np.inf
    return np.argsort(-scores)[:n]


def recommend(customer_number, n=TOP_N):
    customer_row = matrix[customer_number]
    # add up the fingerprints of what they bought, weighted by their matrix values
    profile = (customer_row @ product_vectors).ravel()
    # score every product by how close it is to that profile
    scores = product_vectors @ profile
    # .indices are the column numbers of products they already bought
    scores[customer_row.indices] = -np.inf
    return np.argsort(-scores)[:n]