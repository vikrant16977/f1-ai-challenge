import pandas as pd
import joblib

# Load the trained model
model = joblib.load("../models/random_forest_model.pkl")

# Mock qualifying results for the Emilia-Romagna race
# You'll need to update these with real values later
race_data = [
    {
        "round": 7,
        "track": "Imola",
        "qualifying_position": 13,  # Realistic for Albon
        "constructor": "Williams",
        "driver": "Alex Albon"
    },
    {
        "round": 7,
        "track": "Imola",
        "qualifying_position": 7,   # Strong qualifying for Sainz
        "constructor": "Williams",
        "driver": "Carlos Sainz"
    }
]

df = pd.DataFrame(race_data)

# Predict positions
predicted_positions = model.predict(df)
df["predicted_position"] = predicted_positions

# Round positions to nearest integer (as races have whole-number positions)
df["predicted_position"] = df["predicted_position"].round().astype(int)

# Show predictions
print("🏁 Predicted Finishing Positions:")
print(df[["driver", "constructor", "predicted_position"]])

# F1 scoring system
def position_to_points(pos):
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    return points_map.get(pos, 0)

# Calculate constructors points for Williams only
williams_points = df[df["constructor"] == "Williams"]["predicted_position"].apply(position_to_points).sum()

print(f"\n🔧 Predicted Constructors Points for Williams: {williams_points}")
