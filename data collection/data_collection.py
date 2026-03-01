import pandas as pd
import numpy as np
from nba_api.stats.endpoints import drafthistory, playercareerstats
import time
import os
import sys  

# Force encoding to UTF-8 for the terminal just in case
sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup
if not os.path.exists('data'): 
    os.makedirs('data')

print("--- Fetching Draft History (2010-2022) ---")
try:
    draft = drafthistory.DraftHistory(league_id='00').get_data_frames()[0]
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to NBA API: {e}")
    sys.exit()

# Filter for our specific years
# Endpoint uses 'SEASON' for draft year
players = draft[draft['SEASON'].astype(int).between(2010, 2022)].copy()

print(f"Found {len(players)} players from draft classes 2010-2022.")

# 2. Collection Loop
all_player_seasons = []
total = len(players)

print("Starting collection. This will take ~15 minutes...")

for i, (idx, player) in enumerate(players.iterrows()):
    p_id = player['PERSON_ID']
    p_name = player['PLAYER_NAME']
    d_year = player['SEASON']
    d_pick = player['OVERALL_PICK']
    
    try:
        # Simple text progress bar
        print(f"Progress: {i+1}/{total} | Processing: {p_name}", end="\r")
        
        career = playercareerstats.PlayerCareerStats(
            player_id=p_id,
            per_mode36='PerGame' 
        ).get_data_frames()[0]
        
        if not career.empty:
            career['PLAYER_NAME'] = p_name
            career['DRAFT_YEAR'] = d_year
            career['DRAFT_NUMBER'] = d_pick
            all_player_seasons.append(career)
        
        # 0.6s is the safety limit for the NBA API to prevent IP bans
        time.sleep(0.6)
        
    except Exception as e:
        print(f"\nSkipping {p_name} due to error: {e}")
        time.sleep(2)
        continue

# 3. Save
if all_player_seasons:
    df_raw = pd.concat(all_player_seasons, ignore_index=True)
    df_raw.to_csv('data/raw_seasons.csv', index=False)
    print(f"\n\nSUCCESS: Saved {len(df_raw)} rows to data/raw_seasons.csv")
else:
    print("\n\nFAILED: No data collected.")