# Decision and progress log

| Day | Date | Owner | Decision | Reason | Evidence | Status |
|---|---|---|---|---|---|---|
| 1 | 2026-10-07 | Jagadeesh Aravind Kalyan Shetti | Downloaded Online Retail II and checked both sheets | Confirmed the file opens and has both yearly sheets | Local file, not committed | Done |
| 4 | 2026-10-10 | Jagadeesh Aravind Kalyan Shetti | Stacked both yearly sheets into one table | The time split (DE3) and the matrix (DE4) need each customer's full history in one place; a cell must hold the cumulative quantity across both years | DE1, src/01_load_data.py | Done |
| 4 | 2026-10-10 | Jagadeesh Aravind Kalyan Shetti | Kept Invoice and StockCode as text, not numbers | Both mix digits and letters (cancellations like C489449, variants like 79323P); converting to numbers would destroy the "C" cancellation flag and the variant codes | DE1, src/01_load_data.py | Done |
| 4 | 2026-10-10 | Jagadeesh Aravind Kalyan Shetti | Did no cleaning in DE1; raw.parquet keeps every line | Keeps an untouched "before" picture so DE2's cleaning log has counts to compare against | DE1, outputs/data_summary.csv | Done |
| 4 | 2026-10-10 | Jagadeesh Aravind Kalyan Shetti | Left missing CustomerID (22.77%) and Description (0.41%) as missing, not filled | A customer cannot be invented; lines without a customer are dropped in DE2, not guessed | DE1, outputs/data_profile.csv | Done |