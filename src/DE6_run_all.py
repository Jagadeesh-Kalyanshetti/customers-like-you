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
]

for script in SCRIPTS:
    print(f"Running {script}")
    """
    check=True stops everything at the first script that fails, so a later script never runs on missing or stale files.
    """
    subprocess.run([sys.executable, script], check=True)

print("All scripts finished")