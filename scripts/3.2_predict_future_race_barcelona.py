import pandas as pd
import joblib


model = joblib.load("./models/random_forest_model_barcelona.pkl") 


race_data = [
    {
        "round": 9,
        "track": "Circuit de Barcelona-Catalunya",
        "qualifying_position": 9,  
        "constructor": "Williams",
        "driver": "Alexander Albon"
    },
    {
        "round": 9,
        "track": "Circuit de Barcelona-Catalunya",
        "qualifying_position": 7,  
        "constructor": "Williams",
        "driver": "Carlos Sainz"
    }
]

df = pd.DataFrame(race_data)

# Predict positions
predicted_positions = model.predict(df)
df["predicted_position"] = predicted_positions.round().astype(int)

# Show predictions
print("🏁 Predicted Finishing Positions for Barcelona 2025:")
print(df[["driver", "constructor", "qualifying_position", "predicted_position"]])

# F1 scoring system
def position_to_points(pos):
    points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    return points_map.get(pos, 0)

# Calculate total constructors' points for Williams
williams_points = df["predicted_position"].apply(position_to_points).sum()
print(f"\n🔧 Predicted Constructors Points for Williams: {williams_points}")
