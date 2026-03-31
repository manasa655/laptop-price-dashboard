print("RUNNING TRAIN_MODEL FROM CORRECT FOLDER")

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("laptop_data.csv")
df.drop(columns=["Unnamed: 0"], inplace=True)

# -------------------------
# CLEAN RAM
# -------------------------
df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

# -------------------------
# CLEAN WEIGHT
# -------------------------
df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)

# -------------------------
# EXTRACT CPU GHz
# -------------------------
def extract_ghz(cpu):
    match = re.search(r"(\d+\.\d+)", cpu)
    return float(match.group(1)) if match else np.nan

df["Cpu_GHz"] = df["Cpu"].apply(extract_ghz)

# -------------------------
# EXTRACT STORAGE
# -------------------------
def extract_storage(mem):
    total = 0
    for size in re.findall(r"(\d+)", mem):
        total += int(size)
    return total

df["Storage_GB"] = df["Memory"].apply(extract_storage)

# -------------------------
# CLEAN BRAND
# -------------------------
df = pd.get_dummies(df, columns=["Company"], prefix="Brand")

# Drop rows with missing values
df.dropna(inplace=True)

# -------------------------
# FEATURES & TARGET
# -------------------------
target = "Price"
features = ["Inches", "Ram", "Weight", "Cpu_GHz", "Storage_GB"] + \
           [col for col in df.columns if col.startswith("Brand_")]

X = df[features]
y = df[target]

# -------------------------
# TRAIN/TEST SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODELS
# -------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    metrics[name] = {"r2": r2, "mae": mae}
    print(f"{name} → R²: {r2:.3f}, MAE: {int(mae):,}")

# -------------------------
# SAVE BUNDLE
# -------------------------
joblib.dump(
    {
        "models": models,
        "features": features,
        "metrics": metrics,
        "test_df": X_test,
        "test_target": y_test
    },
    "laptop_price_model.pkl"
)

print("Models saved with metrics and features ✅")
