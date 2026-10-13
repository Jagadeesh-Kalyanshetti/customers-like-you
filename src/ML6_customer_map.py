from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

PROCESSED = Path("data/processed")
OUT = Path("outputs")
SEED = 42
GROUP_OPTIONS = [3, 4, 5, 6]
TOP_PRODUCTS = 5
TOP_COUNTRIES = 3

# I named these after reading segment_profiles.csv (size, products per
# customer, top products, countries). The numbers are k-means group numbers,
# which are the same on every run because the seed is fixed
SEGMENT_NAMES = {
    0: "Occasional small buyers",
    1: "Regular core buyers",
    2: "Largest accounts",
}

# ---- load the train-only matrix and the index files ----
matrix = sp.load_npz(PROCESSED / "sparse_customer_product_matrix.npz")
customer_index = pd.read_parquet(PROCESSED / "customer_index.parquet")
product_index = pd.read_parquet(PROCESSED / "product_index.parquet")

# ---- two scores per customer: one dot on the map (x, y) ----
svd = TruncatedSVD(n_components=2, random_state=SEED)
customer_points = svd.fit_transform(matrix)

# ---- try 3 to 6 groups, keep the one with the best silhouette score ----
choice_rows = []
best_score = -1.0
for n_groups in GROUP_OPTIONS:
    model = KMeans(n_clusters=n_groups, n_init=10, random_state=SEED)
    labels = model.fit_predict(customer_points)
    score = silhouette_score(customer_points, labels)
    choice_rows.append({"n_groups": n_groups, "silhouette": round(score, 4)})
    if score > best_score:
        best_score = score
        best_n_groups = n_groups
        best_labels = labels

# ---- country per customer and name per product (descriptive only) ----
clean = pd.read_parquet(
    PROCESSED / "clean.parquet",
    columns=["CustomerID", "StockCode", "Description", "Country"],
)
country_of = clean.drop_duplicates("CustomerID").set_index("CustomerID")["Country"]
customer_country = country_of.loc[customer_index["CustomerID"]].to_numpy()

name_of = clean.drop_duplicates("StockCode").set_index("StockCode")["Description"]
product_names = name_of.loc[product_index["StockCode"]].astype(str).str.strip().to_numpy()

# ---- describe each group ----
# share of ALL customers who bought each product
share_overall = matrix.getnnz(axis=0) / matrix.shape[0]

profile_rows = []
for group in range(best_n_groups):
    members = np.where(best_labels == group)[0]
    group_matrix = matrix[members]

    # share of THIS group who bought each product, minus the overall share:
    # best sellers are bought by everyone, so they don't make the top list
    share_in_group = group_matrix.getnnz(axis=0) / len(members)
    extra_share = share_in_group - share_overall
    top_columns = np.argsort(-extra_share)[:TOP_PRODUCTS]

    top_product_names = []
    for column_number in top_columns:
        top_product_names.append(product_names[column_number])

    country_shares = pd.Series(customer_country[members]).value_counts(normalize=True)
    top_country_texts = []
    for country, share in country_shares.head(TOP_COUNTRIES).items():
        top_country_texts.append(f"{country} {share:.0%}")

    profile_rows.append({
        "group": group,
        "segment_name": SEGMENT_NAMES.get(group, f"Segment {group + 1}"),
        "customers": len(members),
        "share_of_customers": round(len(members) / matrix.shape[0], 4),
        "avg_products_bought": round(group_matrix.getnnz(axis=1).mean(), 1),
        "top_products": "; ".join(top_product_names),
        "top_countries": "; ".join(top_country_texts),
    })

# ---- the map: one dot per customer, no labels, no IDs (R4) ----
fig, ax = plt.subplots(figsize=(9, 6))
for group in range(best_n_groups):
    in_group = best_labels == group
    segment_name = SEGMENT_NAMES.get(group, f"Segment {group + 1}")
    ax.scatter(
        customer_points[in_group, 0],
        customer_points[in_group, 1],
        s=8,
        alpha=0.6,
        label=f"{segment_name} ({in_group.sum()} customers)",
    )
ax.set_xlabel("Component 1 (overall buying volume)")
ax.set_ylabel("Component 2")
ax.set_title("Customer map: each dot is one customer")
ax.legend()
fig.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "customer_map.png", dpi=150)
pd.DataFrame(profile_rows).to_csv(OUT / "segment_profiles.csv", index=False)
pd.DataFrame(choice_rows).to_csv(OUT / "cluster_choice.csv", index=False)
print("Saved customer map, segment profiles and cluster choice to outputs/")