from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# folders where DE4 saved the matrix and where I write tables and figures
PROCESSED = Path("data/processed")
OUT = Path("outputs")

# the five k values from the brief (ML2)
K_VALUES = [10, 20, 50, 100, 200]

# fixed seed so every run gives the same result (R6)
SEED = 42

# I load only the train-only matrix, so nothing from the held-out pairs is touched (R2)
matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")

"""
square every filled cell, add them up, take the square root
zeros add nothing, so .data (the filled cells only) is enough
"""
matrix_size = np.sqrt((matrix.data ** 2).sum())

# one dict per k goes into this list, and the list becomes the table
rows = []
for k in K_VALUES:
    """
    I fit on the train-only matrix
    random_state keeps the result the same every run (R6)
    """
    svd = TruncatedSVD(n_components=k, random_state=SEED)

    # learns k patterns and returns one short row of k numbers per customer
    customer_factors = svd.fit_transform(matrix)

    """
    size of the rebuilt part is the same recipe:
    square, add, root for the customer factors
    """
    rebuilt_size = np.sqrt((customer_factors ** 2).sum())

    """
    we get the size of what we actually lost using pythagoras:
    whole squared = rebuilt squared + lost squared
    """
    lost_size = np.sqrt(matrix_size ** 2 - rebuilt_size ** 2)

    # lost size divided by whole size is the relative reconstruction error
    rows.append({
        "k": k,
        "relative_error": round(lost_size / matrix_size, 4),
    })

# exist_ok means no error if outputs/ exists; a fresh clone has no outputs/
OUT.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT / "svd_reconstruction_error.csv", index=False)

"""
x axis is k
y axis is relative error
marker="o" shows the actual k values I fitted
"""
errors = pd.DataFrame(rows)

# blank sheet, 6 x 4 inches
plt.figure(figsize=(6, 4))
plt.plot(errors["k"], errors["relative_error"], marker="o")
plt.xlabel("k (number of components)")
plt.ylabel("Relative reconstruction error")
plt.title("Truncated SVD: error versus k")

# light grid lines to read values off the curve
plt.grid(True)

# stops the labels getting cut off
plt.tight_layout()

# dpi=150 gives a 900 x 600 pixel image, enough for the memo and slides
plt.savefig(OUT / "svd_error_curve.png", dpi=150)

print(f"Saved SVD error for {len(rows)} k values to outputs/svd_reconstruction_error.csv and outputs/svd_error_curve.png")