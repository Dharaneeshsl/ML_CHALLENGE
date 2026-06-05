# Technical Architecture

## Problem Type
Traffic Demand Forecasting

**Task Type:** Supervised Regression

**Target Variable:** `demand`

## Data Processing

### Timestamp Parsing
The timestamp field is decomposed into:

- `hour`
- `minute`
- `total_minutes`

### Cyclic Encoding
Temporal periodicity is captured using sine/cosine transforms:

- `hour_sin`, `hour_cos`
- `min_sin`, `min_cos`
- `ts_sin`, `ts_cos`

These features preserve the cyclical nature of time and help models learn periodic patterns.

## Spatial Feature Engineering
Geohash strings are decomposed into hierarchical components:

- `gh3`
- `gh4`
- `gh5`

These capture location granularity at multiple spatial scales.

## Categorical Encoding
Label Encoding is applied to the following categorical fields:

- `geohash`, `gh3`, `gh4`, `gh5`
- `RoadType`, `LargeVehicles`, `Landmarks`, `Weather`

## Aggregation Features
Training-set statistics are generated to capture typical demand patterns.

### Geohash Level
- mean, median, std, min, max demand per geohash

### Timestamp Level
- mean, std demand per timestamp

### Hour Level
- average demand per hour

### Interaction Features
- `geohash × timestamp`
- `gh3 × timestamp`
- `gh4 × timestamp`
- `geohash × hour`
- `gh3 × hour`
- `geohash × day`
- `RoadType × timestamp`
- `NumberofLanes × timestamp`
- `geohash × Weather`

These aggregation features provide location-specific and time-specific traffic behavior signals.

## Model Architecture

### Model 1 — CatBoostRegressor
Configuration:

- iterations = 3000
- depth = 8
- learning_rate = 0.05
- subsample = 0.8

### Model 2 — LightGBMRegressor
Configuration:

- n_estimators = 3000
- num_leaves = 127
- learning_rate = 0.05
- subsample = 0.8

## Validation Strategy
5-Fold Cross Validation

Benefits:

- Reduced variance
- Better generalization
- Robust performance estimation

## Ensemble Strategy
Weighted Averaging

Final Prediction:

Final = 0.50 × CatBoost + 0.50 × LightGBM

## Output Layer
Predictions are clipped to the range [0, 1] to ensure valid demand values.

The final submission file contains:

- Index
- demand

and is exported as `submission.csv`.
