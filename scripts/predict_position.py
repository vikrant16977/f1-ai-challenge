# scripts/predict_position.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load enriched data
df = pd.read_csv("data/f1_enriched_results_2023.csv")

# Drop rows with missing values (e.g., missing qualifying data)
df = df.dropna(subset=["qualifying_position"])

# Features and target
X = df[["round", "track", "qualifying_position", "constructor", "driver"]]
y = df["position"]

# Preprocess: One-hot encode categorical columns
categorical_features = ["track", "constructor", "driver"]
numeric_features = ["round", "qualifying_position"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
], remainder="passthrough")

# Model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"📊 Mean Absolute Error (MAE): {mae:.2f}")

# Sample predictions
sample = X_test.sample(5, random_state=42)
sample_preds = model.predict(sample)
print("\n🧪 Sample Predictions:")
for i, pred in enumerate(sample_preds):
    driver = sample.iloc[i]["driver"]
    quali = sample.iloc[i]["qualifying_position"]
    print(f"{driver} (Qualified P{int(quali)}) ➡ Predicted Finish: P{int(pred)}")
