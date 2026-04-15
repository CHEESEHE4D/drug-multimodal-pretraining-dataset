#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def normalize_text(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def canonical_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def find_best_target_name(df: pd.DataFrame, query_name: str) -> Optional[str]:
    names = df["target_name"].dropna().astype(str).unique().tolist()
    q = canonical_text(query_name)

    for n in names:
        if canonical_text(n) == q:
            return n

    candidates = []
    for n in names:
        cn = canonical_text(n)
        if q in cn or cn in q:
            candidates.append(n)

    if len(candidates) == 1:
        return candidates[0]

    if candidates:
        candidates = sorted(candidates, key=lambda x: abs(len(canonical_text(x)) - len(q)))
        return candidates[0]

    return None


def normalize_activity_type(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = x.strip().upper()
    mapping = {
        "IC50": "IC50",
        "KI": "Ki",
        "KD": "Kd",
        "EC50": "EC50",
    }
    return mapping.get(s, None)


def to_numeric_positive(x: object) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v) and v > 0:
            return v
        return None
    except Exception:
        return None


def assign_label(v: float, active_thr: float, inactive_thr: float) -> Optional[int]:
    if v <= active_thr:
        return 1
    if v >= inactive_thr:
        return 0
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="02_smiles_standardized.csv")
    parser.add_argument("--target_name", required=True, help="目标名称")
    parser.add_argument("--activity_type", required=True, help="IC50 / Ki / Kd / EC50")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--active_threshold_nM", type=float, default=100.0)
    parser.add_argument("--inactive_threshold_nM", type=float, default=1000.0)
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file, low_memory=False)

    required_cols = [
        "source",
        "source_id",
        "molecule_name",
        "smiles_raw",
        "target_name",
        "uniprot_id",
        "activity_type",
        "activity_value",
        "activity_unit",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    work = df.copy()

    # 只保留有原始结构的记录
    work["smiles_raw"] = work["smiles_raw"].map(normalize_text)
    work["target_name"] = work["target_name"].map(normalize_text)
    work["uniprot_id"] = work["uniprot_id"].map(normalize_text)
    work["activity_type_std"] = work["activity_type"].map(normalize_text).map(normalize_activity_type)
    work["activity_unit_std"] = work["activity_unit"].map(normalize_text)
    work["activity_value_nM"] = work["activity_value"].map(to_numeric_positive)

    work = work[
        work["smiles_raw"].notna() &
        work["target_name"].notna() &
        work["activity_type_std"].notna() &
        work["activity_unit_std"].notna() &
        work["activity_value_nM"].notna()
    ].copy()

    # 只保留 nM
    work = work[work["activity_unit_std"] == "nM"].copy()

    matched_name = find_best_target_name(work, args.target_name)
    if matched_name is None:
        raise ValueError(f"找不到 target_name: {args.target_name}")

    sub = work[
        (work["target_name"].astype(str) == matched_name) &
        (work["activity_type_std"] == args.activity_type)
    ].copy()

    # 打标签
    sub["activity_label"] = sub["activity_value_nM"].map(
        lambda v: assign_label(v, args.active_threshold_nM, args.inactive_threshold_nM)
    )
    sub = sub[sub["activity_label"].notna()].copy()
    sub["activity_label"] = sub["activity_label"].astype(int)

    if sub.empty:
        raise ValueError("筛选后没有可用样本")

    # 这里关键：为了复用旧 baseline 脚本，把 smiles_raw 复制到 smiles_std
    sub["smiles_std"] = sub["smiles_raw"]

    # 保留核心列
    keep_cols = [
        "source",
        "source_id",
        "molecule_name",
        "smiles_raw",
        "smiles_std",
        "target_name",
        "uniprot_id",
        "activity_type_std",
        "activity_value_nM",
        "activity_unit_std",
        "activity_label",
    ]
    sub = sub[keep_cols].copy()
    sub = sub.rename(columns={
        "activity_type_std": "activity_type",
        "activity_unit_std": "activity_unit",
    })

    # 不去重，直接切分
    train_valid, test = train_test_split(
        sub,
        test_size=0.2,
        random_state=42,
        stratify=sub["activity_label"]
    )

    train, valid = train_test_split(
        train_valid,
        test_size=0.125,   # 总体 10%
        random_state=42,
        stratify=train_valid["activity_label"]
    )

    sub.to_csv(output_dir / "full.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(output_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    print("matched target:", matched_name)
    print("full :", sub.shape)
    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test :", test.shape)
    print("active:", int((sub["activity_label"] == 1).sum()))
    print("inactive:", int((sub["activity_label"] == 0).sum()))


if __name__ == "__main__":
    main()