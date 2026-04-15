# Drug Multimodal Pretraining Dataset

A project for multimodal drug data collection, standardization, benchmark construction, and baseline evaluation for molecular learning tasks.

## Overview

This project focuses on building a standardized drug-related dataset for multimodal molecular pretraining and benchmark evaluation. The current work mainly covers:

- drug-related data collection and integration
- molecular structure standardization
- molecular descriptor extraction
- drug-like filtering
- activity/target task construction
- baseline model evaluation
- raw vs processed comparison
- random split vs scaffold split comparison

The dataset and pipeline are intended to support fairer and more reproducible evaluation for molecular machine learning tasks.

## Data Sources

The current version mainly uses the following public resources:

- **ChEMBL**
- **BindingDB**
- **DrugBank** (planned/partially prepared for annotation expansion)

## Main Processing Pipeline

The project currently includes the following stages:

1. **Data collection and integration**
   - collect molecular structure and bioactivity-related data
   - integrate records from multiple databases

2. **Molecular preprocessing**
   - SMILES standardization
   - duplicate removal
   - descriptor calculation
   - drug-like filtering

3. **Activity/target task construction**
   - extract valid activity records
   - normalize activity type and unit
   - convert activity values into supervised labels
   - build benchmark-ready molecular activity tasks

4. **Baseline evaluation**
   - Morgan fingerprint generation
   - Logistic Regression
   - Random Forest
   - random split and scaffold split evaluation

## Current Results

The project has completed:

- large-scale molecular data preprocessing
- descriptor table construction
- activity/target benchmark task construction
- first-round baseline experiments on representative targets
- raw vs processed comparison
- random split vs scaffold split comparison

Representative tasks include:

- Acetylcholinesterase (IC50)
- Epidermal growth factor receptor (IC50)
- Coagulation factor X (Ki)
- Dihydrofolate reductase (IC50)
- Neuraminidase (IC50)

## Repository Structure

```text
drug-multimodal-pretraining-dataset/
├── README.md
├── .gitignore
├── src/
│   ├── process_molecule_data.py
│   ├── build_activity_tasks.py
│   ├── make_single_task_split.py
│   ├── make_raw_single_task_split.py
│   ├── make_scaffold_single_task_split.py
│   ├── run_baseline_morgan.py
│   ├── collect_baseline_results.py
│   ├── compare_raw_vs_processed.py
│   └── merge_all_results_comparison.py
├── docs/
│   ├── project_execution_summary.docx
│   └── research_summary_report.docx
│    
├── results/
│   ├── target_task_summary.csv
│   ├── baseline_results_all.csv
│   ├── baseline_results_test_only.csv
│   ├── raw_vs_processed_comparison.csv
│   └── all_results_comparison.csv
└── figures/
```

## Notes on Data Availability

Raw database files are **not included** in this repository due to file size limitations and database distribution restrictions.

Please download the original datasets from their official sources:

- ChEMBL
- BindingDB
- DrugBank

This repository mainly provides:

- preprocessing scripts
- benchmark construction scripts
- baseline evaluation scripts
- summary results
- documentation

## Environment

Recommended environment:

- Python 3.10+
- pandas
- numpy
- scikit-learn
- rdkit

You may prepare the environment with:

```bash
pip install pandas numpy scikit-learn rdkit
```

## Benchmark Tasks

The benchmark currently focuses on molecular activity prediction tasks.

For each task, the pipeline supports:

- task-specific split generation
- random split evaluation
- scaffold split evaluation
- baseline comparison across tasks and models

## Experimental Settings

Current baseline models:

- Logistic Regression
- Random Forest

Current molecular representation:

- Morgan fingerprints

Evaluation metrics:

- ROC-AUC
- PR-AUC
- F1
- Accuracy
- Balanced Accuracy

## Future Work

Planned extensions include:

- expanding DrugBank-based annotation tables
- adding side-effect related information
- incorporating more multimodal information sources
- evaluating stronger molecular representation models
- improving benchmark design under stricter generalization settings

## Acknowledgment

This repository was developed as part of an undergraduate innovation and entrepreneurship research project on multimodal drug-related molecular data processing and benchmark construction.
