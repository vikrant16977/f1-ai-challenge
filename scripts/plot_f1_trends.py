import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('../data/f1_filtered_results_2023.csv')

# Convert 'date' to datetime for sorting
df['date'] = pd.to_datetime(df['date'])

# Sort by round
df = df.sort_values(by='round')

# 🎯 1. Driver Positions over Time
plt.figure(figsize=(12, 6))
sns.lineplot(data=df[df['driver'].isin(['Alex Albon', 'Carlos Sainz'])],
             x='round', y='position', hue='driver', marker='o')

plt.gca().invert_yaxis()  # 1st place is highest
plt.title('Driver Positions Across 2023 Races')
plt.xlabel('Race Round')
plt.ylabel('Position (Lower is Better)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 2. Williams Points Over the Season
williams_df = df[df['constructor'] == 'Williams']
grouped = williams_df.groupby('round')['points'].sum().reset_index()

plt.figure(figsize=(12, 5))
sns.barplot(data=grouped, x='round', y='points', palette='Blues_d')
plt.title('Williams Points Per Round - 2023')
plt.xlabel('Round')
plt.ylabel('Points')
plt.tight_layout()
plt.show()
