import pandas as pd
import numpy as np

def build_arcs():
    print("Building career arcs...")
    
    # 1. Load data and skip bad lines/duplicate headers
    df = pd.read_csv('data/raw_seasons.csv', on_bad_lines='skip')
    
    # 2. Drop rows that are just repeated headers
    df = df[df['PLAYER_ID'] != 'PLAYER_ID'] 
    
    # 3. Force columns to be numbers so math works
    numeric_cols = ['MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'TOV']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. ROBUST SEASON LOGIC
    def extract_year(val):
        val = str(val).strip()
        if '-' in val: # Handles '2022-23'
            return int(val.split('-')[0])
        elif len(val) >= 4: # Handles '22022' -> 2022
            return int(val[-4:])
        return np.nan

    df['SEASON_START'] = df['SEASON_ID'].apply(extract_year)
    df = df.dropna(subset=['SEASON_START']) # Clean up any weird rows

    df['DRAFT_YEAR'] = pd.to_numeric(df['DRAFT_YEAR'], errors='coerce').astype(int)
    df['CAREER_YEAR'] = (df['SEASON_START'] - df['DRAFT_YEAR'] + 1).astype(int)

    # 5. Filter for Years 1, 2, and 5
    # We only care about rows where the player actually played
    df = df[df['CAREER_YEAR'].isin([1, 2, 5])]

    key_stats = ['PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'MIN', 'TOV']
    rows = []

    # Filter to only players who have Year 1 AND Year 2 (so we can predict)
    player_counts = df.groupby('PLAYER_NAME')['CAREER_YEAR'].nunique()
    valid_players = player_counts[player_counts >= 2].index
    
    print(f"Pivoting data for {len(valid_players)} players...")

    for name in valid_players:
        group = df[df['PLAYER_NAME'] == name]
        d_year = group['DRAFT_YEAR'].iloc[0]
        d_pick = group.get('DRAFT_NUMBER', pd.Series([60])).iloc[0] # Default to 60 if missing
        
        row = {'name': name, 'draft_year': d_year, 'draft_pick': d_pick}
        
        for yr in [1, 2, 5]:
            yr_data = group[group['CAREER_YEAR'] == yr]
            # If multiple teams in one year (traded), take the 'TOT' row or first row
            if len(yr_data) > 1:
                yr_data = yr_data.head(1) 
                
            for stat in key_stats:
                val = yr_data[stat].values[0] if not yr_data.empty else np.nan
                row[f'y{yr}_{stat.lower()}'] = val
        rows.append(row)

    final_df = pd.DataFrame(rows)
    # Important: Drop players who have NO Year 5 data if we're training a model
    # (But keep them if you want to use them for 'Active' predictions later)
    final_df.to_csv('data/career_arcs.csv', index=False)
    print(f"✅ Success! Created career_arcs.csv with {len(final_df)} players.")

if __name__ == "__main__":
    build_arcs()