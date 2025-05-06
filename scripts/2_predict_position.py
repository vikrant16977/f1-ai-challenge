import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

# === Load & clean data ===
df = pd.read_csv("../data/f1_enriched_results_2020_2025.csv");
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

# Fit encoder separately to extract feature names
encoder.fit(X[categorical_features])
encoded_feature_names = encoder.get_feature_names_out(categorical_features)
feature_names = list(encoded_feature_names) + numeric_features

# === Pipeline ===
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# === Train/test split & training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Mean Absolute Error: {mae:.2f}")

# === Sample predictions ===
sample = X_test.sample(5, random_state=42)
sample_preds = model.predict(sample)

print("\n🧪 Sample Predictions:")
for i, pred in enumerate(sample_preds):
    driver = sample.iloc[i]["driver"]
    quali = sample.iloc[i]["qualifying_position"]
    print(f"{driver} (Qualified P{int(quali)}) ➡ Predicted Finish: P{int(pred)}")

# === Feature importance plot ===
rf_model = model.named_steps["regressor"]
importances = rf_model.feature_importances_

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(fi_df["Feature"][:20][::-1], fi_df["Importance"][:20][::-1])
plt.title("🔍 Top 20 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# === Save model to file ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "../models/random_forest_model.pkl")
print("\n✅ Model saved to models/random_forest_model.pkl")
