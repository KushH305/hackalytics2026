import pandas as pd
from nba_api.stats.endpoints import drafthistory, playercareerstats
import time
import os
import sys

# Force UTF-8 for Windows Terminal
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 1. Setup
FILE_NAME = 'data/raw_seasons.csv'
if not os.path.exists('data'): os.makedirs('data')

print("--- Data Collection Resumed ---")

# 2. Get the player list
try:
    draft = drafthistory.DraftHistory(league_id='00').get_data_frames()[0]
    players = draft[draft['SEASON'].astype(int).between(2010, 2022)].copy()
except Exception as e:
    print(f"Connection Error: {e}")
    sys.exit()

processed_ids = []
if os.path.exists(FILE_NAME):
    try:
        existing_df = pd.read_csv(FILE_NAME)
        processed_ids = existing_df['PLAYER_ID'].unique().tolist()
        print(f"Skipping {len(processed_ids)} players already in CSV.")
    except:
        pass

# 3. Collection Loop
total = len(players)
for i, (idx, player) in enumerate(players.iterrows()):
    p_id = player['PERSON_ID']
    p_name = player['PLAYER_NAME']
    
    if p_id in processed_ids:
        continue

    try:
        # Simple print, no special characters, no carriage returns to be safe
        print(f"Processing {i+1}/{total}: {p_name}")
        
        career = playercareerstats.PlayerCareerStats(
            player_id=p_id, 
            per_mode36='PerGame'
        ).get_data_frames()[0]
        
        if not career.empty:
            career['PLAYER_NAME'] = p_name
            career['DRAFT_YEAR'] = player['SEASON']
            career['DRAFT_NUMBER'] = player['OVERALL_PICK']
            
            # Append mode 'a'
            header = not os.path.exists(FILE_NAME)
            career.to_csv(FILE_NAME, mode='a', index=False, header=header)
        
        time.sleep(0.8) # Keep this at 0.8 to avoid the timeout ban again
        
    except Exception as e:
        print(f"Error occurred. Waiting 10 seconds...")
        time.sleep(10)
        continue

print("Done!")