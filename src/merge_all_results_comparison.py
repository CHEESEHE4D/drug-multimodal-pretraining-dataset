#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def load_and_prepare(path: str, version_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False).copy()

    required = {"task", "model", "roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列: {missing}")

    # 只保留 test 结果
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()

    df["task"] = df["task"].astype(str)
    df["model"] = df["model"].astype(str)

    # raw 版任务名去掉 _raw
    if version_name == "random_raw":
        df["task"] = df["task"].str.replace("_raw$", "", regex=True)

    rename_map = {
        "roc_auc": f"roc_auc_{version_name}",
        "pr_auc": f"pr_auc_{version_name}",
        "f1": f"f1_{version_name}",
        "accuracy": f"accuracy_{version_name}",
        "balanced_accuracy": f"balanced_accuracy_{version_name}",
    }
    keep_cols = ["task", "model"] + list(rename_map.keys())
    df = df[keep_cols].rename(columns=rename_map)

    return df


def add_delta(df: pd.DataFrame, left: str, right: str, metric: str) -> None:
    lcol = f"{metric}_{left}"
    rcol = f"{metric}_{right}"
    if lcol in df.columns and rcol in df.columns:
        df[f"{metric}_{left}_minus_{right}"] = df[lcol] - df[rcol]


def main():
    parser = argparse.ArgumentParser(description="合并 random/raw/scaffold 三类 benchmark 结果")
    parser.add_argument("--random_processed", required=True, help="processed 随机切分 test 结果 csv")
    parser.add_argument("--random_raw", required=True, help="raw 随机切分 test 结果 csv")
    parser.add_argument("--scaffold_processed", required=True, help="processed scaffold 切分 test 结果 csv")
    parser.add_argument("--output_file", required=True, help="输出总表路径")
    args = parser.parse_args()

    df_random_processed = load_and_prepare(args.random_processed, "random_processed")
    df_random_raw = load_and_prepare(args.random_raw, "random_raw")
    df_scaffold_processed = load_and_prepare(args.scaffold_processed, "scaffold_processed")

    merged = df_random_processed.merge(df_random_raw, on=["task", "model"], how="outer")
    merged = merged.merge(df_scaffold_processed, on=["task", "model"], how="outer")

    metrics = ["roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy"]

    # processed vs raw
    for m in metrics:
        add_delta(merged, "random_processed", "random_raw", m)

    # scaffold vs random_processed
    for m in metrics:
        add_delta(merged, "scaffold_processed", "random_processed", m)

    # scaffold vs raw
    for m in metrics:
        add_delta(merged, "scaffold_processed", "random_raw", m)

    merged = merged.sort_values(["task", "model"]).reset_index(drop=True)

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    print("Saved:", out)
    print(merged)


if __name__ == "__main__":
    main()