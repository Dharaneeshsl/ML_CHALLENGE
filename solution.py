import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# ── catboost / lightgbm imports ─────────────────────────────
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not found. Install: pip install catboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️  LightGBM not found. Install: pip install lightgbm")

if not HAS_CATBOOST or not HAS_LGB:
    raise SystemExit("❌ Please install both catboost and lightgbm before running.")

SEED   = 42
FOLDS  = 5
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ════════════════════════════════════════════════════════════
print("📂 Loading data …")
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print(f"   train: {train.shape}   test: {test.shape}")

# ════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════
print("⚙️  Engineering features …")

def parse_timestamp(df):
    df = df.copy()
    df["hour"]   = df["timestamp"].apply(lambda x: int(x.split(":")[0]))
    df["minute"] = df["timestamp"].apply(lambda x: int(x.split(":")[1]))
    df["ts_min"] = df["hour"] * 60 + df["minute"]
    # cyclic encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]   / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]   / 24)
    df["min_sin"]   = np.sin(2 * np.pi * df["minute"] / 60)
    df["min_cos"]   = np.cos(2 * np.pi * df["minute"] / 60)
    df["ts_sin"]    = np.sin(2 * np.pi * df["ts_min"] / 1440)
    df["ts_cos"]    = np.cos(2 * np.pi * df["ts_min"] / 1440)
    return df

def geohash_features(df):
    df = df.copy()
    df["gh3"] = df["geohash"].str[:3]
    df["gh4"] = df["geohash"].str[:4]
    df["gh5"] = df["geohash"].str[:5]
    return df

train = parse_timestamp(train)
test  = parse_timestamp(test)
train = geohash_features(train)
test  = geohash_features(test)

# ── Label encode categoricals ────────────────────────────────
CAT_COLS = ["geohash", "gh3", "gh4", "gh5",
            "RoadType", "LargeVehicles", "Landmarks", "Weather"]

encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    combined = pd.concat([train[col].fillna("missing"),
                          test[col].fillna("missing")])
    le.fit(combined)
    train[col + "_enc"] = le.transform(train[col].fillna("missing"))
    test[col  + "_enc"] = le.transform(test[col].fillna("missing"))
    encoders[col] = le

# ── Aggregation / target-encoding style features ─────────────
def add_agg_features(df, ref):
    """All aggregations computed on the full training set (ref)."""
    df = df.copy()

    # geohash-level demand stats
    for stat in ["mean", "median", "std", "min", "max"]:
        m = ref.groupby("geohash")["demand"].agg(stat).rename(f"gh_{stat}")
        df = df.join(m, on="geohash")

    # timestamp-level
    for stat in ["mean", "std"]:
        m = ref.groupby("ts_min")["demand"].agg(stat).rename(f"ts_{stat}")
        df = df.join(m, on="ts_min")

    # hour-level
    m = ref.groupby("hour")["demand"].mean().rename("hour_mean")
    df = df.join(m, on="hour")

    # geohash × ts_min
    m = ref.groupby(["geohash","ts_min"])["demand"].mean().rename("gh_ts_mean")
    df = df.join(m, on=["geohash","ts_min"])

    # gh3 × ts_min
    m = ref.groupby(["gh3","ts_min"])["demand"].mean().rename("gh3_ts_mean")
    df = df.join(m, on=["gh3","ts_min"])

    # gh4 × ts_min
    m = ref.groupby(["gh4","ts_min"])["demand"].mean().rename("gh4_ts_mean")
    df = df.join(m, on=["gh4","ts_min"])

    # geohash × hour
    m = ref.groupby(["geohash","hour"])["demand"].mean().rename("gh_hour_mean")
    df = df.join(m, on=["geohash","hour"])

    # gh3 × hour
    m = ref.groupby(["gh3","hour"])["demand"].mean().rename("gh3_hour_mean")
    df = df.join(m, on=["gh3","hour"])

    # geohash × day
    m = ref.groupby(["geohash","day"])["demand"].mean().rename("gh_day_mean")
    df = df.join(m, on=["geohash","day"])

    # RoadType × ts_min
    m = ref.groupby(["RoadType","ts_min"])["demand"].mean().rename("rt_ts_mean")
    df = df.join(m, on=["RoadType","ts_min"])

    # NumberofLanes × ts_min
    m = ref.groupby(["NumberofLanes","ts_min"])["demand"].mean().rename("lanes_ts_mean")
    df = df.join(m, on=["NumberofLanes","ts_min"])

    # geohash × Weather
    m = ref.groupby(["geohash","Weather"])["demand"].mean().rename("gh_weather_mean")
    df = df.join(m, on=["geohash","Weather"])

    return df

train = add_agg_features(train, train)
test  = add_agg_features(test,  train)

# ── Final feature list ───────────────────────────────────────
FEATURES = [
    "hour", "minute", "ts_min", "day",
    "hour_sin", "hour_cos", "min_sin", "min_cos", "ts_sin", "ts_cos",
    "geohash_enc", "gh3_enc", "gh4_enc", "gh5_enc",
    "RoadType_enc", "NumberofLanes", "LargeVehicles_enc",
    "Landmarks_enc", "Temperature", "Weather_enc",
    "gh_mean", "gh_median", "gh_std", "gh_min", "gh_max",
    "ts_mean", "ts_std", "hour_mean",
    "gh_ts_mean", "gh3_ts_mean", "gh4_ts_mean",
    "gh_hour_mean", "gh3_hour_mean", "gh_day_mean",
    "rt_ts_mean", "lanes_ts_mean", "gh_weather_mean",
]

# fill NaN with column median (from train)
train_medians = train[FEATURES].median()
X_train = train[FEATURES].fillna(train_medians)
y_train = train["demand"].values
X_test  = test[FEATURES].fillna(train_medians)

print(f"   Feature count : {len(FEATURES)}")
print(f"   NaN in X_train: {X_train.isnull().sum().sum()}")
print(f"   NaN in X_test : {X_test.isnull().sum().sum()}")

# ════════════════════════════════════════════════════════════
# 3. MODEL DEFINITIONS
# ════════════════════════════════════════════════════════════
def get_catboost():
    return CatBoostRegressor(
        iterations        = 3000,
        learning_rate     = 0.05,
        depth             = 8,
        l2_leaf_reg       = 3,
        subsample         = 0.8,
        colsample_bylevel = 0.8,
        min_data_in_leaf  = 20,
        loss_function     = "RMSE",
        eval_metric       = "R2",
        random_seed       = SEED,
        thread_count      = -1,
        verbose           = False,
    )

def get_lightgbm():
    return lgb.LGBMRegressor(
        n_estimators       = 3000,
        learning_rate      = 0.05,
        num_leaves         = 127,
        max_depth          = -1,
        min_child_samples  = 20,
        subsample          = 0.8,
        colsample_bytree   = 0.8,
        reg_alpha          = 0.1,
        reg_lambda         = 0.1,
        n_jobs             = -1,
        random_state       = SEED,
        verbose            = -1,
    )

# ════════════════════════════════════════════════════════════
# 4. 5-FOLD CROSS-VALIDATION + OOF PREDICTIONS
# ════════════════════════════════════════════════════════════
print("\n🔁 5-Fold CV training …")
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

X_tr = X_train.values
X_te = X_test.values

oof_cb  = np.zeros(len(X_tr))
oof_lgb = np.zeros(len(X_tr))
pred_cb  = np.zeros(len(X_te))
pred_lgb = np.zeros(len(X_te))

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr), 1):
    Xtr, Xval = X_tr[tr_idx], X_tr[val_idx]
    ytr, yval = y_train[tr_idx], y_train[val_idx]

    # ── CatBoost ──────────────────────────────────────────
    cb = get_catboost()
    cb.fit(
        Xtr, ytr,
        eval_set       = (Xval, yval),
        early_stopping_rounds = 200,
        verbose        = False,
    )
    oof_cb[val_idx]  = cb.predict(Xval)
    pred_cb         += cb.predict(X_te) / FOLDS
    cb_r2 = r2_score(yval, oof_cb[val_idx])

    # ── LightGBM ──────────────────────────────────────────
    lg = get_lightgbm()
    lg.fit(
        Xtr, ytr,
        eval_set              = [(Xval, yval)],
        callbacks             = [lgb.early_stopping(200, verbose=False),
                                 lgb.log_evaluation(-1)],
    )
    oof_lgb[val_idx]  = lg.predict(Xval)
    pred_lgb         += lg.predict(X_te) / FOLDS
    lgb_r2 = r2_score(yval, oof_lgb[val_idx])

    print(f"   Fold {fold}/{FOLDS}  |  CB R²={cb_r2:.5f}  |  LGB R²={lgb_r2:.5f}")

# ════════════════════════════════════════════════════════════
# 5. ENSEMBLE (weighted average: 50% CB + 50% LGB)
# ════════════════════════════════════════════════════════════
CB_WEIGHT  = 0.50
LGB_WEIGHT = 0.50

oof_final  = CB_WEIGHT * oof_cb  + LGB_WEIGHT * oof_lgb
pred_final = CB_WEIGHT * pred_cb + LGB_WEIGHT * pred_lgb

# Clip to valid demand range [0, 1]
pred_final = np.clip(pred_final, 0, 1)

oof_score = max(0, 100 * r2_score(y_train, oof_final))
print(f"\n✅ OOF Ensemble R² Score: {oof_score:.5f}")

# ════════════════════════════════════════════════════════════
# 6. SAVE SUBMISSION
# ════════════════════════════════════════════════════════════
submission = pd.DataFrame({
    "Index":  test["Index"].values,
    "demand": pred_final,
})
submission.to_csv("submission.csv", index=False)
print(f"\n📁 submission.csv saved  →  shape: {submission.shape}")
print(f"   demand range: [{pred_final.min():.6f}, {pred_final.max():.6f}]")
print("\n🏆 Done! Upload submission.csv to the leaderboard.")
