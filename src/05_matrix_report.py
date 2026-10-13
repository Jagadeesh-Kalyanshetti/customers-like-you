from pathlib import Path

import pandas as pd
import scipy.sparse as sp

PROCESSED = Path("data/processed")
OUT = Path("outputs")

# This script starts empty, so I load the matrix that 04 saved instead of rebuilding it.
sparse_customer_product_matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")
n_customers, n_products = sparse_customer_product_matrix.shape

test_pairs = pd.read_parquet(PROCESSED / "test_pairs_final.parquet")

"""
The product count before the filter was never saved, so I count it from train. 
I read only StockCode to keep it fast.
"""
n_products_in_train = pd.read_parquet(PROCESSED / "train.parquet", columns=["StockCode"])["StockCode"].nunique()


"""
MB here means 1,000,000 bytes.
Sparsity is divide first, then subtract from 1.
Sparse MB adds data, indices and indptr, because the matrix needs all three.
"""
report_counts = {
    "products_in_train": n_products_in_train,
    "products_kept": n_products,
    "customers_rows": n_customers,
    "products_columns": n_products,
    "filled_cells": sparse_customer_product_matrix.nnz,
    "sparsity": round(1 - sparse_customer_product_matrix.nnz / (n_customers * n_products), 4),
    "dense_mb": round(n_customers * n_products * 8 / 1e6, 2),
    "sparse_mb": round(
        (
            sparse_customer_product_matrix.data.nbytes
            + sparse_customer_product_matrix.indices.nbytes
            + sparse_customer_product_matrix.indptr.nbytes
        )
        / 1e6,
        2,
    ),
    "heldout_pairs": len(test_pairs),
    "heldout_customers": test_pairs["CustomerID"].nunique(),
}

# Each (label, number) pair from the dict becomes one row of the table.
report_rows = []
for item, value in report_counts.items():
    report_rows.append({"item": item, "value": str(value)})
matrix_report = pd.DataFrame(report_rows)

# exist_ok=True: no error if outputs/ already exists. A fresh clone has no such folder.
OUT.mkdir(parents=True, exist_ok=True)

# index=False: no extra 0, 1, 2... column, since the index holds no real labels.
matrix_report.to_csv(OUT / "matrix_report.csv", index=False)

print(f"Saved matrix report with {len(matrix_report)} rows to {OUT / 'matrix_report.csv'}")