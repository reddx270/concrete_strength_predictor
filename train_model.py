"""
train_model.py
---------------
Trains three XGBoost models on the concrete dataset:
  1. Main regressor   -> point estimate of compressive strength
  2. Lower quantile (0.1) -> lower bound of 80% prediction interval
  3. Upper quantile (0.9) -> upper bound of 80% prediction interval

Run once before launching the Streamlit app:
    python train_model.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import json

DATA_FILE = "Concrete_Data_R1.xlsx"

# ------------------------------------------------------------------
# 1. Load and clean
# ------------------------------------------------------------------
df = pd.read_excel(DATA_FILE)
df.columns = ["Cement", "GGBS", "FlyAsh", "Water", "SP", "CoarseAgg",
              "FineAgg", "Age", "Strength", "Source", "TotalKg"]

before = len(df)
df = df[(df["TotalKg"] >= 2150) & (df["TotalKg"] <= 2600)].copy()
print(f"Cleaned: removed {before - len(df)} rows outside plausible density.")
print(f"Working with {len(df)} rows.")

# ------------------------------------------------------------------
# 2. Feature engineering
#    These ratios encode well-known concrete-mix design relationships
#    (Abrams' law: strength is dominated by water/binder ratio).
# ------------------------------------------------------------------
df["Binder"] = df["Cement"] + df["GGBS"] + df["FlyAsh"]
df["WC_Ratio"] = df["Water"] / df["Cement"]
df["WCM_Ratio"] = df["Water"] / df["Binder"]
df["SCM_Pct"] = (df["GGBS"] + df["FlyAsh"]) / df["Binder"]
df["AggRatio"] = df["CoarseAgg"] / df["FineAgg"]
df["SP_per_Binder"] = df["SP"] / df["Binder"]

FEATURES = ["Cement", "GGBS", "FlyAsh", "Water", "SP", "CoarseAgg",
            "FineAgg", "Age", "Binder", "WC_Ratio", "WCM_Ratio",
            "SCM_Pct", "AggRatio", "SP_per_Binder"]
TARGET = "Strength"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# ------------------------------------------------------------------
# 3. Main model
# ------------------------------------------------------------------
common_params = dict(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining main regressor...")
model = xgb.XGBRegressor(objective="reg:squarederror", **common_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"  Test MAE : {mae:.2f} MPa")
print(f"  Test RMSE: {rmse:.2f} MPa")
print(f"  Test R^2 : {r2:.4f}")

print("Running 5-fold cross-validation...")
cv_scores = cross_val_score(
    xgb.XGBRegressor(objective="reg:squarederror", **common_params),
    X, y,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    scoring="r2", n_jobs=-1)
print(f"  CV R^2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ------------------------------------------------------------------
# 4. Quantile models for prediction interval
# ------------------------------------------------------------------
print("\nTraining quantile models for 80% prediction interval...")
model_lo = xgb.XGBRegressor(
    objective="reg:quantileerror", quantile_alpha=0.1, **common_params)
model_lo.fit(X_train, y_train, verbose=False)

model_hi = xgb.XGBRegressor(
    objective="reg:quantileerror", quantile_alpha=0.9, **common_params)
model_hi.fit(X_train, y_train, verbose=False)

y_lo = model_lo.predict(X_test)
y_hi = model_hi.predict(X_test)
coverage = float(np.mean((y_test >= y_lo) & (y_test <= y_hi)))
avg_width = float(np.mean(y_hi - y_lo))
print(f"  80% PI empirical coverage: {coverage:.3f}")
print(f"  80% PI average width     : {avg_width:.2f} MPa")

# ------------------------------------------------------------------
# 5. Feature importance
# ------------------------------------------------------------------
importance = (pd.DataFrame({"feature": FEATURES,
                            "importance": model.feature_importances_})
              .sort_values("importance", ascending=False))
print("\nFeature importance (top 8):")
print(importance.head(8).to_string(index=False))

# ------------------------------------------------------------------
# 6. Persist artifacts
# ------------------------------------------------------------------
joblib.dump(model, "model_main.joblib")
joblib.dump(model_lo, "model_lo.joblib")
joblib.dump(model_hi, "model_hi.joblib")

raw_features = ["Cement", "GGBS", "FlyAsh", "Water", "SP",
                "CoarseAgg", "FineAgg", "Age"]

metadata = {
    "features": FEATURES,
    "raw_features": raw_features,
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "pi_coverage_80": coverage,
        "pi_avg_width": avg_width,
    },
    "feature_importance": importance.to_dict("records"),
    "n_total": int(len(df)),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "feature_ranges": {
        f: {"min": float(df[f].min()),
            "p05": float(df[f].quantile(0.05)),
            "median": float(df[f].median()),
            "p95": float(df[f].quantile(0.95)),
            "max": float(df[f].max())} for f in raw_features
    },
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved: model_main.joblib, model_lo.joblib, model_hi.joblib, metadata.json")
print("Done.")
