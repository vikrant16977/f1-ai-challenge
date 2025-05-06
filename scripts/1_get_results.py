import requests
import pandas as pd
import time
import os

DRIVERS = ["albon", "sainz"]
TARGET_CONSTRUCTOR = "williams"

all_data = []

for year in range(2020, 2026):  # 2020 to 2025 inclusive
    print(f"\n📅 Processing year {year}")
    for round_num in range(1, 23):  # Up to 22 races (some years may have fewer)
        print(f"Fetching round {round_num} of {year}")
        
        result_url = f"http://ergast.com/api/f1/{year}/{round_num}/results.json"
        res = requests.get(result_url).json()

        try:
            race_info = res['MRData']['RaceTable']['Races'][0]
        except IndexError:
            continue

        race_name = race_info['raceName']
        circuit = race_info['Circuit']['circuitName']
        date = race_info['date']
        results = race_info['Results']

        # Get qualifying results
        qual_url = f"http://ergast.com/api/f1/{year}/{round_num}/qualifying.json"
        qual_res = requests.get(qual_url).json()
        try:
            qual_data = qual_res['MRData']['RaceTable']['Races'][0]['QualifyingResults']
        except (IndexError, KeyError):
            qual_data = []

        for r in results:
            driver_id = r['Driver']['driverId']
            constructor = r['Constructor']['name'].lower()
            position = int(r['position'])

            if driver_id in DRIVERS or TARGET_CONSTRUCTOR in constructor:
                name = f"{r['Driver']['givenName']} {r['Driver']['familyName']}"
                constructor_name = r['Constructor']['name']
                grid_position = next((int(q['position']) for q in qual_data if q['Driver']['driverId'] == driver_id), None)

                all_data.append({
                    "year": year,
                    "round": round_num,
                    "race": race_name,
                    "track": circuit,
                    "date": date,
                    "driver": name,
                    "constructor": constructor_name,
                    "qualifying_position": grid_position,
                    "position": position,
                    "points": float(r['points'])
                })

        time.sleep(1)  # be kind to the API

# Save as DataFrame
df = pd.DataFrame(all_data)

# Ensure output directory exists
os.makedirs("../data", exist_ok=True)

# Save the data
df.to_csv("../data/f1_enriched_results_2020_2025.csv", index=False)
print("✅ Data saved to ../data/f1_enriched_results_2020_2025.csv")
