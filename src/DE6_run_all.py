import subprocess
import sys

"""
The order is fixed: each script reads the files the previous one saved, because scripts don't share variables.
"""
SCRIPTS = [
    "src/DE1_load_data.py",
    "src/DE2_clean_data.py",
    "src/DE3_split_data.py",
    "src/DE4_build_sparse_matrix.py",
    "src/DE5_matrix_report.py",
    "src/ML2_fit_svd.py",
    "src/ML4_evaluate.py",
    "src/ML5_pca_vs_svd_fit.py",
    "src/ML6_customer_map.py",
]

for script in SCRIPTS:
    print(f"Running {script}")
    """
    check=True stops everything at the first script that fails, so a later script never runs on missing or stale files.
    """
    subprocess.run([sys.executable, script], check=True)

print("All scripts finished")