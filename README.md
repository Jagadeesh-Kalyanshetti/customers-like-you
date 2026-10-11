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

## Environment
conda environment: miniproj

## How to run
To be completed in the coming time.