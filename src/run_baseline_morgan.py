#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
第一版 baseline：
只用 molecule_activity_clean.csv 切分出的任务文件，
从 smiles_std 生成 Morgan fingerprint，
进行二分类训练与评估。

支持模型：
- LogisticRegression
- RandomForestClassifier

输入：
某一个任务目录下的
- train.csv
- valid.csv
- test.csv

输出：
- metrics_summary.csv
- test_predictions_lr.csv
- test_predictions_rf.csv

依赖：
pip install pandas numpy scikit-learn rdkit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
)


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_X_y(df: pd.DataFrame, smiles_col: str = "smiles_std", label_col: str = "activity_label") -> Tuple[np.ndarray, np.ndarray]:
    fps = []
    labels = []

    for _, row in df.iterrows():
        smi = row[smiles_col]
        y = row[label_col]
        if pd.isna(smi) or pd.isna(y):
            continue
        try:
            fp = smiles_to_morgan_fp(str(smi))
            fps.append(fp)
            labels.append(int(y))
        except Exception:
            continue

    X = np.array(fps, dtype=np.int8)
    y = np.array(labels, dtype=np.int64)
    return X, y


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def fit_and_eval_lr(X_train, y_train, X_valid, y_valid, X_test, y_test):
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        n_jobs=None
    )
    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_prob >= 0.5).astype(int)
    valid_metrics = evaluate_binary(y_valid, valid_pred, valid_prob)

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = evaluate_binary(y_test, test_pred, test_prob)

    return clf, valid_metrics, test_metrics, test_pred, test_prob


def fit_and_eval_rf(X_train, y_train, X_valid, y_valid, X_test, y_test):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    valid_prob = clf.predict_proba(X_valid)[:, 1]
    valid_pred = (valid_prob >= 0.5).astype(int)
    valid_metrics = evaluate_binary(y_valid, valid_pred, valid_prob)

    test_prob = clf.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = evaluate_binary(y_test, test_pred, test_prob)

    return clf, valid_metrics, test_metrics, test_pred, test_prob


def main():
    parser = argparse.ArgumentParser(description="Run first baseline with Morgan fingerprints")
    parser.add_argument("--task_dir", required=True, help="任务目录，里面应有 train.csv / valid.csv / test.csv")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = task_dir / "train.csv"
    valid_file = task_dir / "valid.csv"
    test_file = task_dir / "test.csv"

    train_df = pd.read_csv(train_file, low_memory=False)
    valid_df = pd.read_csv(valid_file, low_memory=False)
    test_df = pd.read_csv(test_file, low_memory=False)

    X_train, y_train = build_X_y(train_df)
    X_valid, y_valid = build_X_y(valid_df)
    X_test, y_test = build_X_y(test_df)

    print("Train:", X_train.shape, y_train.shape)
    print("Valid:", X_valid.shape, y_valid.shape)
    print("Test :", X_test.shape, y_test.shape)

    rows = []

    # Logistic Regression
    _, valid_metrics_lr, test_metrics_lr, test_pred_lr, test_prob_lr = fit_and_eval_lr(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    rows.append({
        "model": "LogisticRegression",
        "split": "valid",
        **valid_metrics_lr
    })
    rows.append({
        "model": "LogisticRegression",
        "split": "test",
        **test_metrics_lr
    })

    pred_lr_df = test_df.iloc[:len(y_test)].copy()
    pred_lr_df["y_true"] = y_test
    pred_lr_df["y_pred"] = test_pred_lr
    pred_lr_df["y_prob"] = test_prob_lr
    pred_lr_df.to_csv(output_dir / "test_predictions_lr.csv", index=False, encoding="utf-8-sig")

    # Random Forest
    _, valid_metrics_rf, test_metrics_rf, test_pred_rf, test_prob_rf = fit_and_eval_rf(
        X_train, y_train, X_valid, y_valid, X_test, y_test
    )

    rows.append({
        "model": "RandomForest",
        "split": "valid",
        **valid_metrics_rf
    })
    rows.append({
        "model": "RandomForest",
        "split": "test",
        **test_metrics_rf
    })

    pred_rf_df = test_df.iloc[:len(y_test)].copy()
    pred_rf_df["y_true"] = y_test
    pred_rf_df["y_pred"] = test_pred_rf
    pred_rf_df["y_prob"] = test_prob_rf
    pred_rf_df.to_csv(output_dir / "test_predictions_rf.csv", index=False, encoding="utf-8-sig")

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Metrics Summary ===")
    print(metrics_df)


if __name__ == "__main__":
    main()