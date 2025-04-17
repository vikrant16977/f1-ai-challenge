import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load data
df = pd.read_csv('f1_filtered_results_2023.csv')

# Select relevant columns
data = df[['round', 'driver', 'constructor', 'position']].copy()

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(data[['driver', 'constructor']])

encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['driver', 'constructor']))
full_data = pd.concat([data[['round']], encoded_df], axis=1)
target = data['position']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(full_data, target, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print(f"📉 Mean Absolute Error: {mae:.2f}")

# Show sample predictions
sample = X_test.copy()
sample['actual'] = y_test.values
sample['predicted'] = predictions.round(1)
print("\n🔍 Sample Predictions:\n")
print(sample.head())
