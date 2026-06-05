import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'train.csv')
TEST_PATH  = os.path.join(BASE_DIR, 'test.csv')
OUT_PATH   = os.path.join(BASE_DIR, 'submission.csv')

print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train : {train.shape}")
print(f"Test  : {test.shape}")

print("\n" + "=" * 60)
print("STEP 2: Feature Engineering")
print("=" * 60)

def engineer_features(df, geo_stats=None):
    df = df.copy()

    # ── Timestamp → numeric ───────────────────────────────────
    df['hour']            = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute']          = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_of_day_mins']= df['hour'] * 60 + df['minute']

    # Cyclical encoding — captures 23:45 → 0:00 continuity
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['min_sin']  = np.sin(2 * np.pi * df['time_of_day_mins'] / 1440)
    df['min_cos']  = np.cos(2 * np.pi * df['time_of_day_mins'] / 1440)

    # Peak-hour flags
    df['is_morning_peak'] = ((df['hour'] >= 7)  & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
    df['is_night']        = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday']       = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)

    # ── Day ───────────────────────────────────────────────────
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)

    # ── Road features ─────────────────────────────────────────
    df['large_vehicles_bin']  = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['landmarks_bin']       = (df['Landmarks'] == 'Yes').astype(int)
    df['is_highway']          = (df['RoadType'] == 'Highway').astype(int)
    df['is_highcap']          = ((df['NumberofLanes'] >= 4) | (df['RoadType'] == 'Highway')).astype(int)
    df['lane_large_interact'] = df['NumberofLanes'] * df['large_vehicles_bin']

    # ── Temperature imputation ────────────────────────────────
    temp_by_weather   = df.groupby('Weather')['Temperature'].transform('median')
    df['Temperature'] = df['Temperature'].fillna(temp_by_weather)
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    # ── Categorical encoding ──────────────────────────────────
    road_map    = {'Residential': 0, 'Street': 1, 'Highway': 2}
    weather_map = {'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3}
    df['RoadType_enc'] = df['RoadType'].map(road_map).fillna(-1).astype(int)
    df['Weather_enc']  = df['Weather'].map(weather_map).fillna(-1).astype(int)

    # ── Geohash target-encoding ───────────────────────────────
    if geo_stats is not None:
        df = df.merge(geo_stats, on='geohash', how='left')
        geo_cols = ['geo_mean','geo_std','geo_median','geo_min',
                    'geo_max','geo_q25','geo_q75','geo_peak_mean','geo_night_mean']
        for col in geo_cols:
            df[col] = df[col].fillna(df[col].median())

    return df

# ── Build geohash stats from train ───────────────────────────
print("Computing geohash-level demand statistics...")
geo_stats = train.groupby('geohash')['demand'].agg(
    geo_mean   = 'mean',
    geo_std    = 'std',
    geo_median = 'median',
    geo_min    = 'min',
    geo_max    = 'max',
    geo_q25    = lambda x: x.quantile(0.25),
    geo_q75    = lambda x: x.quantile(0.75)
).reset_index()

train['hour_tmp']  = train['timestamp'].apply(lambda x: int(x.split(':')[0]))
peak_demand        = train[train['hour_tmp'].between(7,19)].groupby('geohash')['demand'].mean().reset_index()
peak_demand.columns= ['geohash','geo_peak_mean']
night_demand       = train[(train['hour_tmp']>=22)|(train['hour_tmp']<=5)].groupby('geohash')['demand'].mean().reset_index()
night_demand.columns=['geohash','geo_night_mean']

geo_stats = geo_stats.merge(peak_demand, on='geohash', how='left')
geo_stats = geo_stats.merge(night_demand, on='geohash', how='left')
geo_stats['geo_std']        = geo_stats['geo_std'].fillna(0)
geo_stats['geo_night_mean'] = geo_stats['geo_night_mean'].fillna(geo_stats['geo_mean'])
geo_stats['geo_peak_mean']  = geo_stats['geo_peak_mean'].fillna(geo_stats['geo_mean'])

print("Applying features to train and test...")
train_fe = engineer_features(train, geo_stats)
test_fe  = engineer_features(test,  geo_stats)

print("\n" + "=" * 60)
print("STEP 3: Z-Normalization (StandardScaler)")
print("=" * 60)

FEATURE_COLS = [
    'hour', 'minute', 'time_of_day_mins',
    'hour_sin', 'hour_cos', 'min_sin', 'min_cos',
    'is_morning_peak', 'is_evening_peak', 'is_night', 'is_midday',
    'day', 'day_sin', 'day_cos',
    'NumberofLanes', 'large_vehicles_bin', 'landmarks_bin',
    'is_highway', 'is_highcap', 'lane_large_interact',
    'Temperature', 'RoadType_enc', 'Weather_enc',
    'geo_mean', 'geo_std', 'geo_median', 'geo_min', 'geo_max',
    'geo_q25', 'geo_q75', 'geo_peak_mean', 'geo_night_mean'
]

X_train = train_fe[FEATURE_COLS].values
y_train = train_fe['demand'].values
X_test  = test_fe[FEATURE_COLS].values

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Total features : {len(FEATURE_COLS)}")
print(f"Train shape    : {X_train_scaled.shape}")
print(f"Test shape     : {X_test_scaled.shape}")

print("\n" + "=" * 60)
print("STEP 4: LightGBM — 5-Fold Cross Validation")
print("=" * 60)

lgb_params = {
    'objective'       : 'regression',
    'metric'          : 'rmse',
    'learning_rate'   : 0.05,
    'num_leaves'      : 127,
    'max_depth'       : -1,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq'    : 5,
    'reg_alpha'       : 0.1,
    'reg_lambda'      : 0.1,
    'n_estimators'    : 2000,
    'random_state'    : 42,
    'verbose'         : -1,
    'n_jobs'          : -1,
}

kf          = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds   = np.zeros(len(X_train_scaled))
test_preds  = np.zeros(len(X_test_scaled))
fold_scores = []

X_df      = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
X_test_df = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_df), 1):
    X_tr,  X_val = X_df.iloc[tr_idx], X_df.iloc[val_idx]
    y_tr,  y_val = y_train[tr_idx],   y_train[val_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=500)
        ]
    )

    val_pred            = np.clip(model.predict(X_val), 0, 1)
    oof_preds[val_idx]  = val_pred
    score               = r2_score(y_val, val_pred)
    fold_scores.append(score)
    test_preds         += model.predict(X_test_df) / kf.n_splits

    print(f"  Fold {fold} | R² = {score:.6f} | Best iter = {model.best_iteration_}")

test_preds  = np.clip(test_preds, 0, 1)
oof_r2      = r2_score(y_train, oof_preds)
hack_score  = max(0, 100 * oof_r2)

print(f"\n{'='*60}")
print(f"OOF R²              : {oof_r2:.6f}")
print(f"Estimated Hackathon : {hack_score:.2f} / 100")
print(f"Per-fold R²         : {[round(s,4) for s in fold_scores]}")

# Top features
fi = pd.DataFrame({'feature': FEATURE_COLS, 'importance': model.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
print("\nTop 10 Features:")
print(fi.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("STEP 5: Saving submission.csv")
print("=" * 60)

submission = pd.DataFrame({'Index': test['Index'], 'demand': test_preds})
submission.to_csv(OUT_PATH, index=False)

print(f"Saved → {OUT_PATH}")
print(f"Shape  : {submission.shape}  (must be 41778 x 2)")
print(f"Sample :\n{submission.head()}")
print(f"\nDemand range : {test_preds.min():.4f}  –  {test_preds.max():.4f}")