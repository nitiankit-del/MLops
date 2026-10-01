"""Schema and deterministic cleaning; learned imputation lives inside CV."""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
CATEGORIES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "ca": [0, 1, 2, 3],
    "thal": [3, 6, 7],
}
URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"


def clean(raw):
    """Validate original UCI coding and map num>0 to disease=1."""
    expected = FEATURES + ["num"]
    if list(raw.columns) != expected:
        raise ValueError("Unexpected UCI column order/schema")
    df = raw.mask(raw.eq("?")).apply(pd.to_numeric, errors="raise")
    if df["num"].isna().any() or not df["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Invalid target")
    for c, allowed in CATEGORIES.items():
        if not df[c].dropna().isin(allowed).all():
            raise ValueError(f"Invalid category: {c}")
    if not np.isfinite(df.to_numpy()[~np.isnan(df.to_numpy())]).all():
        raise ValueError("Infinite value")
    if (df[NUMERIC].drop(columns="oldpeak") <= 0).any().any():
        raise ValueError("Non-positive clinical measurement")
    if (df["oldpeak"] < 0).any():
        raise ValueError("Negative oldpeak")
    df["target"] = (df.pop("num") > 0).astype(int)
    return df.drop_duplicates().reset_index(drop=True)


def acquire():
    import io
    import zipfile

    folder = ROOT / "data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    raw_path = folder / "processed.cleveland.data"
    if not raw_path.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            archive = response.read()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            name = next(n for n in z.namelist() if n.endswith("processed.cleveland.data"))
            raw_path.write_bytes(z.read(name))
    expected_hash = "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8"
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(
            "Raw dataset checksum mismatch; verify the UCI source before updating the pin"
        )
    raw = pd.read_csv(raw_path, names=FEATURES + ["num"], na_values="?")
    df = clean(raw)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "heart.csv", index=False)
    manifest = {
        "source": URL,
        "citation": "Janosi et al. (1989), UCI Heart Disease, DOI:10.24432/C52P4X",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "features": FEATURES,
        "missing": df.isna().sum().to_dict(),
        "class_counts": df.target.value_counts().to_dict(),
        "cleaning": "Original categories; num>0; exact duplicates removed; missing retained for fold-local imputation",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return df


if __name__ == "__main__":
    acquire()
