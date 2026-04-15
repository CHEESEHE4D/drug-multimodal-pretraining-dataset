#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="molecule_activity_clean.csv")
    parser.add_argument("--target_name", required=True, help="example_target_name，必须和表里一致")
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

    train_valid, test = train_test_split(
        sub,
        test_size=0.2,
        random_state=42,
        stratify=sub["activity_label"]
    )

    train, valid = train_test_split(
        train_valid,
        test_size=0.125,   # 0.125 of 0.8 = 0.1 total
        random_state=42,
        stratify=train_valid["activity_label"]
    )

    sub.to_csv(output_dir / "full.csv", index=False, encoding="utf-8-sig")
    train.to_csv(output_dir / "train.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(output_dir / "valid.csv", index=False, encoding="utf-8-sig")
    test.to_csv(output_dir / "test.csv", index=False, encoding="utf-8-sig")

    print("full :", sub.shape)
    print("train:", train.shape)
    print("valid:", valid.shape)
    print("test :", test.shape)


if __name__ == "__main__":
    main()