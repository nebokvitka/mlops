import os
import json
import pandas as pd


def test_data_schema():
    data_path = os.getenv("DATA_PATH", "data/raw/heart.csv")
    assert os.path.exists(data_path), f"Data not found: {data_path}"
    df = pd.read_csv(data_path)
    required_cols = {"age", "sex", "cp", "trestbps", "chol",
                     "fbs", "restecg", "thalach", "exang",
                     "oldpeak", "slope", "ca", "thal", "target"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns: {sorted(missing)}"
    assert df["target"].notna().all(), "target contains NaN"
    assert df.shape[0] >= 50, "Too few rows"


def test_artifacts_exist():
    assert os.path.exists("model.pkl"), "model.pkl not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("confusion_matrix.png"), "confusion_matrix.png not found"


def test_quality_gate_f1():
    threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    f1 = float(metrics["f1"])
    assert f1 >= threshold, f"Quality Gate failed: f1={f1:.4f} < {threshold:.2f}"