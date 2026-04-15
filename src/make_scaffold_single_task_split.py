#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def get_bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def scaffold_split(df: pd.DataFrame, smiles_col: str = "smiles_std"):
    """
    简单 scaffold split:
    - 先按 scaffold 分组
    - 再按 scaffold 组大小从大到小分配到 train/valid/test
    - 目标比例 0.7 / 0.1 / 0.2
    """
    df = df.copy()
    df["scaffold"] = df[smiles_col].map(get_bemis_murcko_scaffold)
    df = df[df["scaffold"] != ""].copy()

    scaffold_to_indices = defaultdict(list)
    for idx, scaf in zip(df.index, df["scaffold"]):
        scaffold_to_indices[scaf].append(idx)

    scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

    n_total = len(df)
    train_target = int(0.7 * n_total)
    valid_target = int(0.1 * n_total)

    train_idx, valid_idx, test_idx = [], [], []

    for group in scaffold_groups:
        if len(train_idx) + len(group) <= train_target:
            train_idx.extend(group)
        elif len(valid_idx) + len(group) <= valid_target:
            valid_idx.extend(group)
        else:
            test_idx.extend(group)

    train = df.loc[train_idx].copy()
    valid = df.loc[valid_idx].copy()
    test = df.loc[test_idx].copy()

    return train, valid, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="molecule_activity_clean.csv")
    parser.add_argument("--target_name", required=True, help="目标名称")
    parser.add_argument("--activity_type", required=True, help="IC50 / Ki / Kd / EC50")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    args = parser.parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file, low_memory=False)

    sub = df[
        (df["target_name"] == args.target_name) &
        (df["activity_type"] == args.activity_type)
    ].copy()

    sub = sub.dropna(subset=["smiles_std", "activity_label"]).copy()
    sub["activity_label"] = sub["activity_label"].astype(int)

    if sub.empty:
        raise ValueError("筛选后没有可用样本")

    train, valid, test = scaffold_split(sub, smiles_col="smiles_std")

    sub.to_csv(output_dir / "full.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(output_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    print("full :", sub.shape)
    print("train:", train.shape, "active=", int((train["activity_label"] == 1).sum()), "inactive=", int((train["activity_label"] == 0).sum()))
    print("valid:", valid.shape, "active=", int((valid["activity_label"] == 1).sum()), "inactive=", int((valid["activity_label"] == 0).sum()))
    print("test :", test.shape, "active=", int((test["activity_label"] == 1).sum()), "inactive=", int((test["activity_label"] == 0).sum()))


if __name__ == "__main__":
    main()