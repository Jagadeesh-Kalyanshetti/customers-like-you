# Customers Like You

Mini-project: PCA or SVD for a retail recommendation problem.
Solo project: I hold both the Data Engineer and ML Engineer roles.

## Success targets:
1. The compressed data is much smaller than the full matrix
2. The nightly job runs fast
3. Hit rate at 10 beats the best-seller baseline on held-out purchases
4. The customer map shows clear groups marketing can name

## Data
Source: UCI Online Retail II (`online_retail_II.xlsx`, two sheets: 2009-2010 and 2010-2011).

- Rows: 1,067,371 invoice lines, 8 columns
- Date range: 1 Dec 2009 to 9 Dec 2011
- Raw distinct values: 5,942 customers, 5,305 stock codes
- Missing: `CustomerID` on 22.77% of lines (243,007); `Description` on 0.41% (4,382)
- Full tables: `outputs/data_profile.csv` and `outputs/data_summary.csv`

### Cleaning (DE2)
- Rows before cleaning: 1,067,371; after: 802,632; removed in total: 264,739
- Rules applied in this order: cancellations, non-positive Quantity, non-positive Price, StockCodes that do not start with 5 digits, lines without a CustomerID
- Rows removed per rule: `outputs/cleaning_log.csv`
- Cleaned table: `data/processed/clean.parquet` (not committed, it contains customer IDs)

### Split (DE3)
- Test period is the last 2 months; the cutoff is computed from the latest invoice date: 2011-10-09 12:50:00
- Train: 684,069 invoice lines before the cutoff. Test period: 118,563 lines on or after it
- Held-out pairs are products a customer bought in the test period and never bought in train (R2)
- Pairs are distinct (customer, product) combinations; rows are invoice lines
- Funnel of pairs: 95,018 bought in the test period, then 63,252 after removing pairs already bought in train (31,766, about 33% of 95,018), then 49,423 after removing customers with no train history (13,829, about 22% of 63,252), then 45,753 after removing products never seen in train (3,670, about 7% of 49,423)
- Final held-out set: 45,753 pairs from 1,916 customers (about 24 new products per customer)
- In train, 684,069 lines shrink to 417,719 distinct pairs, so about 39% of lines are repeat purchases of a product the customer already bought
- Counts per step: `outputs/split_summary.csv`
- `data/processed/train.parquet` and `test_pairs.parquet` are not committed (they contain customer IDs)

### Matrix (DE4)
- Built from train only (R2): one row per customer, one column per product, stored sparse
- Each cell holds log1p of the customer's summed quantity for that product
- Product filter: kept products bought by at least 10 customers in train. Of 4,529 products in train, 3,687 are kept (842 dropped, about 19% of 4,529)
- Matrix: 5,473 customers x 3,687 products with 413,927 filled cells. The 417,719 train pairs shrink to 413,927 after the filter (3,792 dropped)
- Held-out pairs after the filter: 45,753 shrink to 44,727 (1,026 dropped, about 2% of 45,753) because their product has no column or their customer has no row. Test customers: 1,916 shrink to 1,913
- Files in `data/processed/` (not committed, they contain customer IDs): `sparse_customer_product_matrix.npz`, `customer_index.parquet`, `product_index.parquet`, `test_pairs_final.parquet`. The position in each index file is the matrix row or column number

### Matrix report (DE5)
- Sparsity: the matrix is 97.95% empty (0.9795 of the 5,473 x 3,687 cells are zero)
- Memory (MB = 1,000,000 bytes): a dense matrix would need about 161.43 MB; the sparse one needs about 4.99 MB (data, indices and indptr arrays together)
- Full report: `outputs/matrix_report.csv`, written by `src/DE5_matrix_report.py`

### Pipeline (DE6)
- `src/DE6_run_all.py` runs the other scripts in `src/` in a fixed order: `DE1_load_data.py`, `DE2_clean_data.py`, `DE3_split_data.py`, `DE4_build_sparse_matrix.py`, `DE5_matrix_report.py`, `ML2_fit_svd.py`, `ML4_evaluate.py`
- `ML3_recommend.py` is not in the run, because it saves nothing; ML4 repeats its logic for all customers at once
- Scripts don't share variables; each one reads the files the previous one saved in `data/processed/`
- The brief calls this one-command rebuild `build_matrix.py`; mine is `DE6_run_all.py`, and it also runs the model scripts
- Model scripts are named by task (`ML2_...` and so on)

### Method choice (ML1)
- Main model: Truncated SVD, decided on 2026-10-13 before any model was fitted; PCA is run at the same k in ML5 for comparison
- Reason: the matrix is 97.95% empty, SVD works on it as it is, and one fit gives customer and product factors together
- Full note: `outputs/decision_note.md`

### SVD fit and choice of k (ML2)
- Truncated SVD is fitted on the train-only matrix for k = 10, 20, 50, 100 and 200 (seed 42)
- Relative reconstruction error = size of what is lost / size of the whole matrix (Frobenius norm). It is computed from the customer factors with Pythagoras (whole squared = rebuilt squared + lost squared), so the dense matrix is never rebuilt
- The error falls steadily from 0.8761 at k = 10 to 0.6395 at k = 200, with no sharp elbow: even at k = 200 about 64% of the matrix is still lost. This is reconstruction error only; it says nothing yet about recommendation quality (ML4)
- Chosen k = 200 (error 0.6395, the lowest of the five). The curve has no elbow, so I took the lowest error. The stored factors are 200 x (5,473 + 3,687) x 8 bytes, about 14.66 MB, against 161.43 MB dense. Lower reconstruction error does not prove better recommendations; ML4 tests that. k was fixed before any held-out scoring and will not be changed after it
- Files: `outputs/svd_reconstruction_error.csv` and `outputs/svd_error_curve.png`, written by `src/ML2_fit_svd.py`

### Recommender (ML3)
- `src/ML3_recommend.py` fits Truncated SVD on the train-only matrix at k = 200 (seed 42) and defines two functions. It saves no file, because the brief's ML3 output is the recommendation function itself
- Each product gets a fingerprint of 200 numbers (a row of `svd.components_.T`), scaled to length 1, so multiplying two fingerprints gives their cosine similarity
- `similar_products(product_number)` returns the 10 products most similar to one product
- `recommend(customer_number)` mixes the fingerprints of the products a customer bought (weighted by their matrix values) into one taste profile, scores every product against it, removes products they already bought, and returns the top 10
- Both functions return matrix column numbers; `product_index.parquet` turns them into StockCodes. Customer numbers are matrix row positions, not customer IDs
- Quality is not tested here; the hit rate against the best-seller baseline comes in ML4

### Evaluation and timing (ML4)
- `src/ML4_evaluate.py` refits SVD at k = 200 (seed 42) and times the nightly job in three parts: the fit, the "also bought" box for all 3,687 products, and the top 10 new products for all 5,473 customers. The held-out pairs are loaded only after the timed part (R2)
- Hit rate at 10 = share of test customers for whom at least one of their 10 suggestions is a held-out product. Test set: 1,913 customers and 44,727 held-out pairs
- Baseline (R3): the 10 products with the most distinct buyers in train, skipping products the customer already bought. Same customers, same held-out pairs
- Result: SVD hit rate 0.2148, best-seller baseline 0.2687. **SVD loses to the baseline by about 5.4 percentage points**, so success target 3 is not met. One test set and one run, so this is a measurement on this split, not a proven difference
- Timing (single run on my MacBook, seconds): fit 0.42, all products 0.81, all customers 1.2, about 2.4 in total. Timings change slightly from run to run; the hit rates do not
- k stayed at 200, because it was fixed in ML2 before any held-out scoring. Changing it after seeing these results would leak the test set into the model choice
- Hypotheses for the loss, not tested: the held-out products are new to each customer and popular products are bought by many people, and suggestions built from a customer's own history lean towards niche products
- File: `outputs/recommender_results.csv` (counts, rates and timings only, no customer IDs), written by `src/ML4_evaluate.py`

## Environment
Python 3.14.8, in a conda environment named miniproj (conda 26.7.1; conda is optional, any Python environment works).

Packages and exact versions are in `requirements.txt`: matplotlib, numpy, openpyxl, pandas, pyarrow, scikit-learn, scipy.

## How to run
1. Use Python 3.14 (I used 3.14.8) and install the packages: `pip install -r requirements.txt`
2. Put the raw file `online_retail_II.xlsx` in `data/raw/` (it is not committed; download it from https://archive.ics.uci.edu/dataset/502/online+retail+ii)
3. From the project root, run `python src/DE6_run_all.py`
4. This rebuilds everything from the raw file: the loaded and cleaned data, the time split, the sparse matrix, the matrix report, the SVD error table and curve, and the hit rate and timing table. Tables and figures are written to `outputs/` and the files containing customer IDs stay in `data/processed/`.
5. Random seeds are fixed at 42 wherever randomness is used.

Run steps for the remaining model scripts (ML5, ML6) will be added here as they are written.