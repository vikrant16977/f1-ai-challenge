import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# === Load & clean data ===
df = pd.read_csv("../data/f1_enriched_results_2020_2025.csv")
df = df.dropna(subset=["qualifying_position"])

# === Features & target ===
X = df[["round", "track", "qualifying_position", "constructor", "driver"]]
y = df["position"]

categorical_features = ["track", "constructor", "driver"]
numeric_features = ["round", "qualifying_position"]

# === Preprocessing ===
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer([
    ("cat", encoder, categorical_features)
], remainder="passthrough")

# Fit encoder separately to extract feature names (optional visualization)
encoder.fit(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
feature_names = list(encoded_feature_names) + numeric_features

# === Pipeline ===
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === Train/test split & training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Mean Absolute Error: {mae:.2f}")

# === Predict on full data ===
full_preds = model.predict(X)
df["predicted_position"] = full_preds.round().astype(int)

print("\n🏁 Predictions on All Data:")
for i in range(len(df)):
    row = df.iloc[i]
    print(f"{row['driver']} (Qualified P{int(row['qualifying_position'])}) ➡ Predicted Finish: P{row['predicted_position']}")

# === Save model to file ===
os.makedirs("../models", exist_ok=True)
joblib.dump(model, "../models/random_forest_model.pkl")
print("\n✅ Model saved to models/random_forest_model.pkl")
