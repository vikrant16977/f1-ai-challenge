import requests
import pandas as pd

TARGET_DRIVERS = ['Alex Albon', 'Carlos Sainz']
TARGET_CONSTRUCTOR = 'Williams'

def fetch_season_data(season):
    all_data = []

    for round_number in range(1, 23):  # F1 usually has ~22 races
        print(f"📦 Fetching Round {round_number}...")

        url = f"https://ergast.com/api/f1/{season}/{round_number}/results.json"
        response = requests.get(url)

        if response.status_code != 200:
            print(f"❌ Skipping Round {round_number} - No data")
            continue

        data = response.json()
        races = data['MRData']['RaceTable']['Races']

        if not races:
            continue

        race = races[0]
        race_name = race['raceName']
        race_date = race['date']

        for result in race['Results']:
            driver = result['Driver']
            constructor = result['Constructor']['name']
            driver_name = f"{driver['givenName']} {driver['familyName']}"

            if driver_name in TARGET_DRIVERS or constructor == TARGET_CONSTRUCTOR:
                all_data.append({
                    'season': season,
                    'round': round_number,
                    'race': race_name,
                    'date': race_date,
                    'driver': driver_name,
                    'constructor': constructor,
                    'position': int(result['position']),
                    'points': float(result['points']),
                    'status': result['status']
                })

    return pd.DataFrame(all_data)

# Fetch 2023 data
df = fetch_season_data(2023)

# Save to CSV
df.to_csv('f1_filtered_results_2023.csv', index=False)
print("✅ Data saved to f1_filtered_results_2023.csv")
