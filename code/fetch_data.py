"""Download the IBM HR dataset from OpenML (ID 43897) into data/raw.

The employee-promotion dataset must be downloaded manually from Kaggle (login required):
https://www.kaggle.com/datasets/arashnic/hr-ana  ->  extract train.csv into data/raw/promotion_train.csv
"""
from pathlib import Path

from sklearn.datasets import fetch_openml

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

ds = fetch_openml(data_id=43897, as_frame=True, parser="liac-arff")
out = RAW / "openml_43897_ibm-employee-performance.csv"
ds.frame.to_csv(out, index=False)
print(f"saved {out.name}: {ds.frame.shape}")

if not (RAW / "promotion_train.csv").exists():
    print("NOTE: data/raw/promotion_train.csv not found - download it from Kaggle (see docstring).")
