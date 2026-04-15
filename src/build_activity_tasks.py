#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从 02_smiles_standardized.csv 构建活性/靶点任务表

输入：
- 02_smiles_standardized.csv

输出：
- molecule_activity_clean.csv
- target_task_summary.csv

功能：
1. 提取活性/靶点相关记录
2. 只保留标准化成功且有 smiles_std 的记录
3. 标准化 target_id
4. 标准化 activity_type
5. 清洗 activity_value
6. 统一单位到 nM（当前脚本默认只保留 nM）
7. 计算 pActivity = 9 - log10(activity_value_nM)
8. 构造二分类标签：
   active <= active_threshold_nM
   inactive >= inactive_threshold_nM
   中间灰区丢弃
9. 输出每个 target 的任务统计

用法示例：
& "C:\\Users\\asus\\anaconda3\\python.exe" "C:\\Users\\asus\\Desktop\\毕业论文\\大创2025--阶段1\\build_activity_tasks.py" `
  --input_file "C:\\Users\\asus\\Desktop\\毕业论文\\大创2025--阶段1\\数据处理结果\\processed\\02_smiles_standardized.csv" `
  --output_dir "C:\\Users\\asus\\Desktop\\毕业论文\\大创2025--阶段1\\数据处理结果\\activity_tasks"

注意：
- 这份脚本默认优先做 IC50 / Ki / Kd / EC50
- 默认只保留单位为 nM 的记录
- 默认 active <= 100 nM, inactive >= 1000 nM
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "activity_task_build.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )


def normalize_text(x: object) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def normalize_target_name(name: Optional[str]) -> Optional[str]:
    """
    简单清洗 target_name，去掉多余空格。
    这里只做轻量规范化，不做复杂实体归一。
    """
    if name is None:
        return None
    s = re.sub(r"\s+", " ", name.strip())
    return s if s else None


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


def build_target_id_std(uniprot_id: Optional[str], target_name: Optional[str]) -> Optional[str]:
    """
    优先用 UniProt ID。
    如果没有，再用清洗后的 target_name。
    """
    if uniprot_id is not None:
        return f"UNIPROT::{uniprot_id}"
    if target_name is not None:
        return f"NAME::{target_name}"
    return None


def to_numeric_positive(x: object) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v) and v > 0:
            return v
        return None
    except Exception:
        return None


def calc_pactivity_nm(value_nm: float) -> float:
    """
    pActivity = 9 - log10(value in nM)
    例如：
    1 nM -> 9
    10 nM -> 8
    100 nM -> 7
    1000 nM -> 6
    """
    return 9.0 - math.log10(value_nm)


def main():
    parser = argparse.ArgumentParser(description="构建活性/靶点任务表")
    parser.add_argument("--input_file", required=True, help="02_smiles_standardized.csv 路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument(
        "--keep_activity_types",
        nargs="+",
        default=["IC50", "Ki", "Kd", "EC50"],
        help="保留的 activity_type"
    )
    parser.add_argument(
        "--allowed_units",
        nargs="+",
        default=["nM"],
        help="允许的活性单位，默认仅保留 nM"
    )
    parser.add_argument(
        "--active_threshold_nM",
        type=float,
        default=100.0,
        help="active 阈值，默认 <=100 nM"
    )
    parser.add_argument(
        "--inactive_threshold_nM",
        type=float,
        default=1000.0,
        help="inactive 阈值，默认 >=1000 nM"
    )
    parser.add_argument(
        "--min_total_per_target",
        type=int,
        default=200,
        help="一个 target 至少多少条已标注记录才视为可用"
    )
    parser.add_argument(
        "--min_active_per_target",
        type=int,
        default=50,
        help="一个 target 至少多少条 active 记录才视为可用"
    )
    parser.add_argument(
        "--min_inactive_per_target",
        type=int,
        default=50,
        help="一个 target 至少多少条 inactive 记录才视为可用"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("开始读取输入文件: %s", args.input_file)
    df = pd.read_csv(args.input_file, low_memory=False)
    logging.info("输入表 shape=%s", df.shape)

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
        "smiles_std",
        "inchikey_std",
        "std_success",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}")

    # 1. 只保留标准化成功且 smiles_std 不为空的记录
    work = df.copy()
    work = work[work["std_success"] == True].copy()
    work["smiles_std"] = work["smiles_std"].map(normalize_text)
    work["inchikey_std"] = work["inchikey_std"].map(normalize_text)
    work = work[work["smiles_std"].notna()].copy()
    logging.info("标准化成功且有 smiles_std 的记录数: %d", len(work))

    # 2. 清洗靶点与活性字段
    work["target_name_std"] = work["target_name"].map(normalize_text).map(normalize_target_name)
    work["uniprot_id_std"] = work["uniprot_id"].map(normalize_text)
    work["activity_type_std"] = work["activity_type"].map(normalize_text).map(normalize_activity_type)
    work["activity_unit_std"] = work["activity_unit"].map(normalize_text)
    work["activity_value_std"] = work["activity_value"].map(to_numeric_positive)

    # 3. 只保留有 target / activity 的记录
    work["target_id_std"] = [
        build_target_id_std(u, t)
        for u, t in zip(work["uniprot_id_std"], work["target_name_std"])
    ]

    before_filter = len(work)
    work = work[
        work["target_id_std"].notna() &
        work["activity_type_std"].notna() &
        work["activity_unit_std"].notna() &
        work["activity_value_std"].notna()
    ].copy()
    logging.info("过滤后保留有 target 和 activity 的记录: %d -> %d", before_filter, len(work))

    # 4. 只保留指定 activity_type 和单位
    keep_types = set(args.keep_activity_types)
    allowed_units = set(args.allowed_units)

    work = work[
        work["activity_type_std"].isin(keep_types) &
        work["activity_unit_std"].isin(allowed_units)
    ].copy()
    logging.info("保留指定 activity_type/单位 后记录数: %d", len(work))

    # 5. 计算 pActivity
    work["activity_value_nM"] = work["activity_value_std"]
    work["pActivity"] = work["activity_value_nM"].map(calc_pactivity_nm)

    # 6. 构建二分类标签
    # active <= active_threshold
    # inactive >= inactive_threshold
    # 中间区间丢弃
    active_thr = args.active_threshold_nM
    inactive_thr = args.inactive_threshold_nM

    if active_thr > inactive_thr:
        raise ValueError("active_threshold_nM 不能大于 inactive_threshold_nM")

    def assign_label(v: float) -> Optional[int]:
        if v <= active_thr:
            return 1
        if v >= inactive_thr:
            return 0
        return None

    work["activity_label"] = work["activity_value_nM"].map(assign_label)
    labeled = work[work["activity_label"].notna()].copy()
    labeled["activity_label"] = labeled["activity_label"].astype(int)

    logging.info("获得二分类标签的记录数: %d", len(labeled))
    logging.info("其中 active=%d, inactive=%d",
                 int((labeled["activity_label"] == 1).sum()),
                 int((labeled["activity_label"] == 0).sum()))

    # 7. 去掉完全重复的实验记录
    dedup_cols = [
        "inchikey_std",
        "target_id_std",
        "activity_type_std",
        "activity_value_nM",
        "activity_label",
    ]
    before_dedup = len(labeled)
    labeled = labeled.drop_duplicates(subset=dedup_cols).copy()
    logging.info("去重后记录数: %d -> %d", before_dedup, len(labeled))

    # 8. 输出清洗后的活性记录表
    activity_out_cols = [
        "source",
        "source_id",
        "molecule_name",
        "smiles_raw",
        "smiles_std",
        "inchikey_std",
        "target_name_std",
        "uniprot_id_std",
        "target_id_std",
        "activity_type_std",
        "activity_value_nM",
        "activity_unit_std",
        "pActivity",
        "activity_label",
    ]
    activity_clean = labeled[activity_out_cols].copy()
    activity_clean = activity_clean.rename(columns={
        "target_name_std": "target_name",
        "uniprot_id_std": "uniprot_id",
        "activity_type_std": "activity_type",
        "activity_value_nM": "activity_value_nM",
        "activity_unit_std": "activity_unit",
    })

    activity_path = output_dir / "molecule_activity_clean.csv"
    activity_clean.to_csv(activity_path, index=False, encoding="utf-8-sig")
    logging.info("已输出: %s, shape=%s", activity_path, activity_clean.shape)

    # 9. 生成 target 任务统计
    summary = (
        activity_clean
        .groupby(["target_id_std", "activity_type"], dropna=False)
        .agg(
            n_total=("activity_label", "size"),
            n_active=("activity_label", lambda x: int((x == 1).sum())),
            n_inactive=("activity_label", lambda x: int((x == 0).sum())),
            n_unique_molecules=("inchikey_std", pd.Series.nunique),
            example_target_name=("target_name", "first"),
            example_uniprot_id=("uniprot_id", "first"),
        )
        .reset_index()
    )

    summary["usable_for_benchmark"] = (
        (summary["n_total"] >= args.min_total_per_target) &
        (summary["n_active"] >= args.min_active_per_target) &
        (summary["n_inactive"] >= args.min_inactive_per_target)
    )

    summary = summary.sort_values(
        by=["usable_for_benchmark", "n_total", "n_active", "n_inactive"],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    summary_path = output_dir / "target_task_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logging.info("已输出: %s, shape=%s", summary_path, summary.shape)

    # 10. 输出一个简短汇总
    stats_text = [
        "# 活性/靶点任务构建汇总",
        f"- input_rows: {len(df)}",
        f"- standardized_rows_with_smiles: {len(work)}",
        f"- labeled_rows_before_dedup: {before_dedup}",
        f"- labeled_rows_after_dedup: {len(activity_clean)}",
        f"- active_threshold_nM: <= {active_thr}",
        f"- inactive_threshold_nM: >= {inactive_thr}",
        f"- usable_targets: {int(summary['usable_for_benchmark'].sum())}",
        f"- total_target_tasks: {len(summary)}",
    ]
    summary_md = output_dir / "activity_task_summary.md"
    summary_md.write_text("\n".join(stats_text), encoding="utf-8")
    logging.info("已输出: %s", summary_md)

    logging.info("全部完成。")


if __name__ == "__main__":
    main()