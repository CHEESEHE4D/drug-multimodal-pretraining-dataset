#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比 raw vs processed 两套 baseline 结果

输入：
- processed_results_file: baseline_results_test_only.csv
- raw_results_file: baseline_results_test_only.csv

输出：
- raw_vs_processed_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Compare raw vs processed benchmark results")
    parser.add_argument("--processed_results_file", required=True, help="processed 的 baseline_results_test_only.csv")
    parser.add_argument("--raw_results_file", required=True, help="raw 的 baseline_results_test_only.csv")
    parser.add_argument("--output_file", required=True, help="输出比较结果 csv")
    args = parser.parse_args()

    processed = pd.read_csv(args.processed_results_file, low_memory=False)
    raw = pd.read_csv(args.raw_results_file, low_memory=False)

    processed["version"] = "processed"
    raw["version"] = "raw"
    raw["task"] = raw["task"].astype(str).str.replace("_raw$", "", regex=True)
    key_cols = ["task", "model"]

    metrics = ["roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy"]

    merged = processed.merge(
        raw,
        on=key_cols,
        suffixes=("_processed", "_raw")
    )

    for m in metrics:
        if f"{m}_processed" in merged.columns and f"{m}_raw" in merged.columns:
            merged[f"{m}_delta"] = merged[f"{m}_processed"] - merged[f"{m}_raw"]

    out = Path(args.output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")

    print("Saved:", out)
    print(merged)


if __name__ == "__main__":
    main()