#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整提交版药物分子标准化处理脚本

功能：
1. 读取 ChEMBL 和 BindingDB 原始数据
2. 提取并统一关键字段
3. 进行 SMILES 标准化
4. 生成标准化 InChIKey
5. 去重
6. 计算分子描述符（QED, LogP, MW, TPSA, SAS 等）
7. 进行类药分子筛选
8. 绘制核密度曲线图

用法示例：
python process_molecule_data.py \
    --chembl raw_data/chembl/chembl_36_chemreps.txt.gz \
    --bindingdb raw_data/bindingdb/BindingDB_All_202604.tsv \
    --output_dir output

依赖：
pip install pandas numpy matplotlib scipy rdkit-pypi

说明：
1. SAS 分数优先尝试从 RDKit Contrib 的 sascorer 导入
2. 如果当前环境没有 sascorer，会自动把 SAS 记为 NaN，并继续运行
3. KDE 图依赖 scipy
"""

from __future__ import annotations

import argparse
import gzip
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, QED
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ========= 尝试导入 scipy 用于 KDE =========
try:
    from scipy.stats import gaussian_kde
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ========= 尝试导入 SAS scorer =========
SAS_AVAILABLE = False
sascorer = None

try:
    # 常见位置 1：conda / RDKit contrib
    from rdkit.Contrib.SA_Score import sascorer  # type: ignore
    SAS_AVAILABLE = True
except Exception:
    try:
        # 常见位置 2：用户把 sascorer.py 放在同目录或 PYTHONPATH
        import sascorer  # type: ignore
        SAS_AVAILABLE = True
    except Exception:
        SAS_AVAILABLE = False


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "processing.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def read_table_auto(path: str) -> pd.DataFrame:
    """
    自动读取 txt/tsv/csv/gz 文件。
    """
    path_obj = Path(path)
    suffixes = "".join(path_obj.suffixes).lower()

    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        sep = ","
    else:
        sep = "\t"

    logging.info("读取文件: %s", path)
    df = pd.read_csv(path, sep=sep, compression="infer", low_memory=False)
    logging.info("读取完成: %s, shape=%s", path, df.shape)
    return df


def normalize_col_name(col: str) -> str:
    col = col.strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def extract_chembl(chembl_path: str) -> pd.DataFrame:
    """
    从 ChEMBL chemreps 文件提取核心字段。
    预期文件：chembl_36_chemreps.txt.gz
    常见列：
    chembl_id, canonical_smiles, standard_inchi, standard_inchi_key
    """
    df = read_table_auto(chembl_path)
    df = normalize_columns(df)

    chembl_id_col = find_first_existing(df, ["chembl_id", "molecule_chembl_id"])
    smiles_col = find_first_existing(df, ["canonical_smiles", "smiles"])
    inchi_col = find_first_existing(df, ["standard_inchi", "inchi"])
    inchikey_col = find_first_existing(df, ["standard_inchi_key", "inchi_key", "inchikey"])

    required = [chembl_id_col, smiles_col]
    if any(c is None for c in required):
        raise ValueError(
            f"ChEMBL 文件缺少必要列。检测到列: {list(df.columns)[:30]}"
        )

    out = pd.DataFrame({
        "source": "ChEMBL",
        "source_id": df[chembl_id_col].astype(str),
        "molecule_name": pd.NA,
        "smiles_raw": df[smiles_col].astype(str),
        "inchikey_raw": df[inchikey_col].astype(str) if inchikey_col else pd.NA,
        "inchi_raw": df[inchi_col].astype(str) if inchi_col else pd.NA,
        "target_name": pd.NA,
        "uniprot_id": pd.NA,
        "activity_type": pd.NA,
        "activity_value": pd.NA,
        "activity_unit": pd.NA,
        "structure_3d_flag": pd.NA,
        "reference_db_version": "chembl_36",
    })

    logging.info("ChEMBL 核心字段提取完成: shape=%s", out.shape)
    return out


def extract_bindingdb(bindingdb_path: str) -> pd.DataFrame:
    """
    从 BindingDB 全量 TSV 提取核心字段。
    常见列名有版本差异，因此做宽松匹配。
    """
    df = read_table_auto(bindingdb_path)
    df = normalize_columns(df)

    # 常见列候选
    binding_id_col = find_first_existing(df, [
        "bindingdb_reactant_set_id",
        "bindingdb_id",
        "reactant_set_id",
        "bindingdb_monomerid",
    ])

    smiles_col = find_first_existing(df, [
        "ligand_smiles",
        "smiles",
        "canonical_smiles",
    ])

    inchikey_col = find_first_existing(df, [
        "ligand_inchi_key",
        "inchi_key",
        "inchikey",
    ])

    molecule_name_col = find_first_existing(df, [
        "ligand_name",
        "ligand",
        "compound_name",
    ])

    target_name_col = find_first_existing(df, [
        "target_name",
        "target_name_assigned_by_curator_or_data_source",
        "target",
    ])

    uniprot_col = find_first_existing(df, [
        "uniprot_swissprot_primary_id_of_target_chain",
        "uniprot_id",
        "uniprot_accession",
        "primary_uniprot_id",
    ])

    ki_col = find_first_existing(df, ["ki_nm", "ki_n_m", "ki"])
    kd_col = find_first_existing(df, ["kd_nm", "kd_n_m", "kd"])
    ic50_col = find_first_existing(df, ["ic50_nm", "ic50_n_m", "ic50"])
    ec50_col = find_first_existing(df, ["ec50_nm", "ec50_n_m", "ec50"])

    required = [binding_id_col, smiles_col]
    if any(c is None for c in required):
        raise ValueError(
            f"BindingDB 文件缺少必要列。检测到列: {list(df.columns)[:50]}"
        )

    out = pd.DataFrame({
        "source": "BindingDB",
        "source_id": df[binding_id_col].astype(str),
        "molecule_name": df[molecule_name_col].astype(str) if molecule_name_col else pd.NA,
        "smiles_raw": df[smiles_col].astype(str),
        "inchikey_raw": df[inchikey_col].astype(str) if inchikey_col else pd.NA,
        "inchi_raw": pd.NA,
        "target_name": df[target_name_col].astype(str) if target_name_col else pd.NA,
        "uniprot_id": df[uniprot_col].astype(str) if uniprot_col else pd.NA,
        "activity_type": pd.NA,
        "activity_value": pd.NA,
        "activity_unit": pd.NA,
        "structure_3d_flag": pd.NA,
        "reference_db_version": Path(bindingdb_path).stem,
    })

    # 统一把 Ki/Kd/IC50/EC50 展开成长表
    activity_frames = []

    def make_activity(sub_df: pd.DataFrame, value_col: Optional[str], act_type: str) -> Optional[pd.DataFrame]:
        if value_col is None:
            return None
        tmp = sub_df.copy()
        tmp["activity_type"] = act_type
        tmp["activity_value"] = pd.to_numeric(df[value_col], errors="coerce")
        tmp["activity_unit"] = "nM"
        tmp = tmp[tmp["activity_value"].notna()]
        return tmp

    for col_name, act_name in [
        (ki_col, "Ki"),
        (kd_col, "Kd"),
        (ic50_col, "IC50"),
        (ec50_col, "EC50"),
    ]:
        frame = make_activity(out, col_name, act_name)
        if frame is not None and not frame.empty:
            activity_frames.append(frame)

    if activity_frames:
        out = pd.concat(activity_frames, ignore_index=True)
    else:
        logging.warning("BindingDB 未检测到 Ki/Kd/IC50/EC50 数值列，将保留基础信息但无活性值。")

    logging.info("BindingDB 核心字段提取完成: shape=%s", out.shape)
    return out


def build_standardizer():
    """
    创建 RDKit 标准化工具。
    """
    largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()
    uncharger = rdMolStandardize.Uncharger()
    normalizer = rdMolStandardize.Normalizer()
    reionizer = rdMolStandardize.Reionizer()
    return largest_fragment_chooser, uncharger, normalizer, reionizer


def standardize_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str], Optional[str], Optional[str]]:
    """
    返回:
    mol_std, smiles_std, inchi_std, inchikey_std
    """
    if smiles is None or pd.isna(smiles):
        return None, None, None, None

    smiles = str(smiles).strip()
    if not smiles:
        return None, None, None, None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None, None

        Chem.SanitizeMol(mol)

        largest_fragment_chooser, uncharger, normalizer, reionizer = build_standardizer()

        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)
        mol = uncharger.uncharge(mol)
        mol = largest_fragment_chooser.choose(mol)

        Chem.SanitizeMol(mol)

        smiles_std = Chem.MolToSmiles(mol, canonical=True)
        inchi_std = Chem.MolToInchi(mol)
        inchikey_std = Chem.InchiToInchiKey(inchi_std)
        return mol, smiles_std, inchi_std, inchikey_std

    except Exception:
        return None, None, None, None


def calc_sas(mol: Chem.Mol) -> Optional[float]:
    if not SAS_AVAILABLE:
        return np.nan
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return np.nan


def calc_descriptors(mol: Chem.Mol) -> Dict[str, Any]:
    """
    计算分子描述符。
    """
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "QED": QED.qed(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
        "RingCount": Lipinski.RingCount(mol),
        "HeavyAtomCount": Lipinski.HeavyAtomCount(mol),
        "SAS": calc_sas(mol),
    }


def plot_kde(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    """
    使用 matplotlib + scipy 绘制 KDE。
    若 scipy 不可用，则退化为直方图。
    """
    data = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(data) < 2:
        logging.warning("数据太少，无法绘图: %s", output_path.name)
        return

    plt.figure(figsize=(8, 5))

    if SCIPY_AVAILABLE and len(np.unique(data)) > 1:
        xs = np.linspace(data.min(), data.max(), 500)
        kde = gaussian_kde(data)
        ys = kde(xs)
        plt.plot(xs, ys)
        plt.fill_between(xs, ys, alpha=0.25)
        plt.ylabel("Density")
    else:
        plt.hist(data, bins=40, density=True, alpha=0.7)
        plt.ylabel("Frequency Density")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logging.info("图已保存: %s", output_path)


def save_summary(stats: Dict[str, Any], output_path: Path) -> None:
    lines = ["# 数据处理汇总\n"]
    for k, v in stats.items():
        lines.append(f"- {k}: {v}")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("汇总说明已保存: %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="完整提交版药物分子标准化处理脚本")
    parser.add_argument("--chembl", required=True, help="ChEMBL chemreps 文件路径，如 chembl_36_chemreps.txt.gz")
    parser.add_argument("--bindingdb", required=True, help="BindingDB 主 TSV 文件路径")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--mw_min", type=float, default=200.0, help="类药筛选 MW 下限")
    parser.add_argument("--mw_max", type=float, default=600.0, help="类药筛选 MW 上限")
    parser.add_argument("--logp_max", type=float, default=5.0, help="类药筛选 LogP 上限")
    parser.add_argument("--tpsa_max", type=float, default=140.0, help="类药筛选 TPSA 上限")
    parser.add_argument("--qed_min", type=float, default=0.30, help="类药筛选 QED 下限")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    figures_dir = output_dir / "figures"
    docs_dir = output_dir / "docs"

    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)

    logging.info("========== 开始处理 ==========")
    logging.info("SAS 可用: %s", SAS_AVAILABLE)
    logging.info("SciPy 可用: %s", SCIPY_AVAILABLE)

    # 1. 提取原始核心字段
    chembl_df = extract_chembl(args.chembl)
    bindingdb_df = extract_bindingdb(args.bindingdb)
 
    raw_merged = pd.concat([chembl_df, bindingdb_df], ignore_index=True)
    raw_merged.to_csv(processed_dir / "01_raw_merged.csv", index=False, encoding="utf-8-sig")
    logging.info("已输出 01_raw_merged.csv, shape=%s", raw_merged.shape)

    # 2. SMILES 标准化
    std_records = []
    success_count = 0

    for idx, row in raw_merged.iterrows():
        mol, smiles_std, inchi_std, inchikey_std = standardize_smiles(row["smiles_raw"])
        ok = mol is not None

        record = row.to_dict()
        record["smiles_std"] = smiles_std
        record["inchi_std"] = inchi_std
        record["inchikey_std"] = inchikey_std
        record["std_success"] = ok
        std_records.append(record)

        if ok:
            success_count += 1

        if (idx + 1) % 50000 == 0:
            logging.info("SMILES 标准化进度: %d / %d", idx + 1, len(raw_merged))

    standardized_df = pd.DataFrame(std_records)
    standardized_df.to_csv(processed_dir / "02_smiles_standardized.csv", index=False, encoding="utf-8-sig")
    logging.info("已输出 02_smiles_standardized.csv, shape=%s", standardized_df.shape)

    # 3. 去掉标准化失败记录
    valid_df = standardized_df[standardized_df["std_success"] == True].copy()

    # 4. 去重：优先 inchikey_std，其次 smiles_std
    valid_df["dedup_key"] = valid_df["inchikey_std"].fillna(valid_df["smiles_std"])
    before_dedup = len(valid_df)
    dedup_df = valid_df.drop_duplicates(subset=["dedup_key"]).copy()
    after_dedup = len(dedup_df)
    dedup_df.to_csv(processed_dir / "03_deduplicated_dataset.csv", index=False, encoding="utf-8-sig")
    logging.info("已输出 03_deduplicated_dataset.csv, shape=%s", dedup_df.shape)

    # 5. 计算分子描述符
    desc_records = []

    for idx, row in dedup_df.iterrows():
        mol, _, _, _ = standardize_smiles(row["smiles_std"])
        if mol is None:
            continue

        desc = calc_descriptors(mol)
        record = row.to_dict()
        record.update(desc)
        desc_records.append(record)

        if (idx + 1) % 20000 == 0:
            logging.info("描述符计算进度: %d / %d", idx + 1, len(dedup_df))

    desc_df = pd.DataFrame(desc_records)
    desc_df.to_csv(processed_dir / "04_molecular_descriptors.csv", index=False, encoding="utf-8-sig")
    logging.info("已输出 04_molecular_descriptors.csv, shape=%s", desc_df.shape)

    # 6. 类药筛选
    filtered_df = desc_df[
        (desc_df["MW"] >= args.mw_min) &
        (desc_df["MW"] <= args.mw_max) &
        (desc_df["LogP"] <= args.logp_max) &
        (desc_df["TPSA"] <= args.tpsa_max) &
        (desc_df["QED"] >= args.qed_min)
    ].copy()

    filtered_df["druglike_flag"] = True
    filtered_df.to_csv(processed_dir / "05_druglike_filtered_dataset.csv", index=False, encoding="utf-8-sig")
    logging.info("已输出 05_druglike_filtered_dataset.csv, shape=%s", filtered_df.shape)

    # 7. 绘图
    plot_kde(desc_df["MW"], "MW Distribution", "MW", figures_dir / "mw_kde.png")
    plot_kde(desc_df["QED"], "QED Distribution", "QED", figures_dir / "qed_kde.png")
    plot_kde(desc_df["LogP"], "LogP Distribution", "LogP", figures_dir / "logp_kde.png")
    plot_kde(desc_df["TPSA"], "TPSA Distribution", "TPSA", figures_dir / "tpsa_kde.png")

    # 8. 输出汇总说明
    stats = {
        "raw_merged_rows": len(raw_merged),
        "smiles_standardization_success_rows": success_count,
        "valid_rows_after_standardization": len(valid_df),
        "rows_before_deduplication": before_dedup,
        "rows_after_deduplication": after_dedup,
        "descriptor_rows": len(desc_df),
        "druglike_rows": len(filtered_df),
        "sas_available": SAS_AVAILABLE,
        "scipy_available_for_kde": SCIPY_AVAILABLE,
        "mw_filter": f"{args.mw_min} <= MW <= {args.mw_max}",
        "logp_filter": f"LogP <= {args.logp_max}",
        "tpsa_filter": f"TPSA <= {args.tpsa_max}",
        "qed_filter": f"QED >= {args.qed_min}",
    }
    save_summary(stats, docs_dir / "data_processing_summary.md")

    logging.info("========== 全部完成 ==========")


if __name__ == "__main__":
    main()