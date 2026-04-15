#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
汇总 benchmark_results 下多个任务目录中的 metrics_summary.csv

输出：
- baseline_results_all.csv
- baseline_results_test_only.csv

目录结构示例：
benchmark_results/
├── acetylcholinesterase_ic50/
│   └── metrics_summary.csv
├── egfr_ic50/
│   └── metrics_summary.csv
├── coagulation_factor_x_ki/
│   └── metrics_summary.csv
...

依赖：
pip install pandas
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Collect baseline results from multiple tasks")
    parser.add_argument("--results_root", required=True, help="benchmark_results 根目录")
    parser.add_argument("--output_dir", required=True, help="汇总输出目录")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for task_dir in sorted(results_root.iterdir()):
        if not task_dir.is_dir():
            continue

        metrics_file = task_dir / "metrics_summary.csv"
        if not metrics_file.exists():
            continue

        try:
            df = pd.read_csv(metrics_file, low_memory=False)
        except Exception as e:
            print(f"Skip {metrics_file}: {e}")
            continue

        df["task"] = task_dir.name
        rows.append(df)

    if not rows:
        raise FileNotFoundError("没有找到任何 metrics_summary.csv")

    all_df = pd.concat(rows, ignore_index=True)

    # 调整列顺序
    preferred_cols = [
        "task", "model", "split",
        "roc_auc", "pr_auc", "f1", "accuracy", "balanced_accuracy"
    ]
    existing_cols = [c for c in preferred_cols if c in all_df.columns]
    other_cols = [c for c in all_df.columns if c not in existing_cols]
    all_df = all_df[existing_cols + other_cols]

    all_out = output_dir / "baseline_results_all.csv"
    all_df.to_csv(all_out, index=False, encoding="utf-8-sig")

    test_df = all_df[all_df["split"] == "test"].copy()
    test_out = output_dir / "baseline_results_test_only.csv"
    test_df.to_csv(test_out, index=False, encoding="utf-8-sig")

    print("Saved:", all_out)
    print("Saved:", test_out)
    print("\n=== All Results ===")
    print(all_df)
    print("\n=== Test Only ===")
    print(test_df)


if __name__ == "__main__":
    main()