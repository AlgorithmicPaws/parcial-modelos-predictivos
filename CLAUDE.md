# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a predictive models course project (semestre 9, parcial) containing two datasets for building and evaluating machine learning models.

## Datasets

### `Metro_Interstate_Traffic_Volume.csv` (~48,204 rows)
- **Task:** Regression — predict `traffic_volume`
- **Features:** `holiday`, `temp`, `rain_1h`, `snow_1h`, `clouds_all`, `weather_main`, `weather_description`, `date_time`
- **Notes:** Contains datetime features requiring time-based feature engineering; `holiday` and weather fields are categorical

### `dataset-uci.csv` (~319 rows)
- **Task:** Classification — predict `Gallstone Status` (binary: 0/1)
- **Features:** Demographics (Age, Gender), comorbidities (CAD, Hypothyroidism, Hyperlipidemia, DM), body composition metrics (BMI, TBW, ECW, ICW, fat ratios, muscle mass, etc.), and lab values (Glucose, Cholesterol, LDL, HDL, Triglyceride, AST, ALT, ALP, Creatinine, GFR, CRP, HGB, Vitamin D)
- **Notes:** Small dataset — prefer cross-validation over a fixed train/test split; many correlated body composition features suggest dimensionality reduction may help

## Typical Workflow

When building notebooks or scripts for this project:

```bash
# Install common ML dependencies if needed
pip install pandas scikit-learn matplotlib seaborn jupyter xgboost

# Launch Jupyter
jupyter notebook
```

## Conventions

- Use Python with pandas, scikit-learn, and matplotlib/seaborn
- For the traffic dataset, extract time features from `date_time` (hour, day of week, month, is_weekend) before modeling
- For the gallstone dataset, handle class imbalance if present (check `Gallstone Status` distribution)
- Prefer pipelines (`sklearn.pipeline.Pipeline`) to avoid data leakage between train/test splits
