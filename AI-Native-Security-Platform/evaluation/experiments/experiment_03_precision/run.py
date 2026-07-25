"""Experiment 3: Anomaly Detection Precision, Recall, F1, and FPR Sensitivity Benchmark."""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from evaluation.datasets.generator import generate_evaluation_dataset
from prototype.edge.detector.model import EdgeAnomalyDetector

def run_precision_experiment() -> pd.DataFrame:
    df_data = generate_evaluation_dataset(num_samples=3000, anomaly_ratio=0.10, seed=42)
    features = df_data[["flow_duration", "packet_count", "byte_rate", "syn_ratio", "port_entropy"]].values
    labels = df_data["label"].values

    detector = EdgeAnomalyDetector(n_estimators=100, random_state=42)
    # Fit on normal subset
    normal_subset = features[labels == 0]
    detector.fit_baseline(normal_subset)

    # Compute raw anomaly scores for all samples
    raw_scores = []
    for f in features:
        _, score = detector.predict_anomaly(f)
        raw_scores.append(score)
    raw_scores = np.array(raw_scores)

    thresholds = np.linspace(0.3, 0.85, 12)
    results = []

    for th in thresholds:
        preds = (raw_scores >= th).astype(int)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results.append({
            "decision_threshold": round(th, 3),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        })

    return pd.DataFrame(results)
