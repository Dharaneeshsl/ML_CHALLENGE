# Gridlock Hackathon 2.0 — Traffic Demand Prediction

## Overview
This solution predicts traffic demand using an ensemble of CatBoost and LightGBM regression models.

The pipeline combines:

- Feature Engineering
- Geohash-Based Aggregations
- Time-Based Feature Extraction
- 5-Fold Cross Validation
- CatBoost Regressor
- LightGBM Regressor
- Weighted Ensemble

## Installation
Install required packages:

```bash
pip install catboost lightgbm scikit-learn pandas numpy scipy
```

## Execution
Place the following files in the same directory:

- `train.csv`
- `test.csv`
- `solution.py`

Run:

```bash
python solution.py
```

## Output
The script produces `submission.csv` with the following columns:

- Index
- demand

Expected shape: 41778 × 2

## Machine Learning Pipeline

1. Load train and test datasets.
2. Generate temporal features from timestamps.
3. Generate geohash hierarchy features.
4. Create aggregation statistics from training data.
5. Encode categorical variables.
6. Train CatBoost using 5-Fold CV.
7. Train LightGBM using 5-Fold CV.
8. Average predictions from both models.
9. Generate `submission.csv`.

## Models

### CatBoostRegressor

- Iterations: 3000
- Depth: 8
- Learning Rate: 0.05

### LightGBMRegressor

- Trees: 3000
- Num Leaves: 127
- Learning Rate: 0.05

## Ensemble
Final Prediction:

0.5 × CatBoost Prediction + 0.5 × LightGBM Prediction

## Reproducibility
The submission is generated entirely from the provided `train.csv` and `test.csv` files and `solution.py`.
No external datasets or precomputed answers are used.
