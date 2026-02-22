import pandas as pd
import joblib
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
from nba_api.stats.static import players
import json




# 1. Base Directory logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
UPDATED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'updated_models')

# 2. DATA PATHS (Matches the 'data' folder in your image)
DATA_PATH = os.path.join(UPDATED_MODELS_DIR, 'data', 'final_player_predictions.csv')
CLUSTER_STATS_PATH = os.path.join(UPDATED_MODELS_DIR, 'data', 'cluster_stats.csv')
FEATURE_IMPORTANCE_PATH = os.path.join(UPDATED_MODELS_DIR, 'data', 'feature_importance_path1.csv')
TEST_P1_PATH = os.path.join(UPDATED_MODELS_DIR, 'data', 'test_predictions_path1.csv')
TEST_P2_PATH = os.path.join(UPDATED_MODELS_DIR, 'data', 'test_predictions_path2.csv')

# 3. MODEL PATHS (Matches the nested 'models/pathX' structure)
# Inside 'models', you have 'path1' and 'path2' folders
MODEL_PATH_P1 = os.path.join(UPDATED_MODELS_DIR, 'models', 'path1', 'final_model.pkl')
MODEL_PATH_P2 = os.path.join(UPDATED_MODELS_DIR, 'models', 'path2', 'final_model.pkl')

# Scaler and KMeans are directly in the 'models' folder (not in path1/path2)
SCALER_PATH = os.path.join(UPDATED_MODELS_DIR, 'models', 'scaler.pkl')
KMEANS_PATH = os.path.join(UPDATED_MODELS_DIR, 'models', 'kmeans_model.pkl')

# --- ARCHEOTYPE DICTIONARY ---
ARCHETYPE_MAP = {
    0: {"name": "Floor General", "desc": "High AST/TOV ratio. Primary offensive engine and playmaker."},
    1: {"name": "Paint Protector", "desc": "Interior force. High REB and FG% near the rim."},
    2: {"name": "3-and-D Wing", "desc": "Floor spacers who provide defensive versatility and perimeter shooting."},
    3: {"name": "Offensive Engine", "desc": "High usage scorers. Primary offensive options for their teams."},
    4: {"name": "Rotation Specialist", "desc": "Value depth players with niche roles (bench scoring/defense)."}
}

# --- Add these to your paths section in utils.py ---
PATH_COMPARISON_JSON = os.path.join(UPDATED_MODELS_DIR, 'data', 'path_comparison.json')
METADATA_P1_JSON = os.path.join(UPDATED_MODELS_DIR, 'data', 'model_metadata_path1.json')
METADATA_P2_JSON = os.path.join(UPDATED_MODELS_DIR, 'data', 'model_metadata_path2.json')

# Add to your existing imports
import plotly.graph_objects as go

# Add this new constant (after ARCHETYPE_MAP)
STAT_GLOSSARY = {
    'Y2 PPG': """
**Year 2 Points Per Game**

**Calculation:** Total points scored divided by games played in sophomore season.

**Basketball Context:** Shows current scoring volume. Used to distinguish 'Offensive Engines' 
from 'Role Players'. High values indicate the player is already a primary scoring option.
    """,
    
    'Y2 RPG': """
**Year 2 Rebounds Per Game**

**Calculation:** Total rebounds (offensive + defensive) divided by games played.

**Basketball Context:** High values identify 'Paint Protectors' who dominate the glass. 
Rebounding is the most consistent skill—players who rebound well in Year 2 usually remain 
elite rebounders throughout their career.
    """,
    
    'Y2 APG': """
**Year 2 Assists Per Game**

**Calculation:** Total assists divided by games played.

**Basketball Context:** The primary identifier for 'Floor Generals' and high-IQ playmakers. 
Assist numbers correlate strongly with ball-handling responsibility and offensive role.
    """,
    
    'Y2 3P%': """
**Year 2 Three-Point Percentage**

**Calculation:** 3-pointers made divided by 3-pointers attempted.

**Basketball Context:** Differentiates '3-and-D Wings' from interior specialists. 
High efficiency here is critical for modern NBA spacing. Threshold for "shooter" is typically 36%+.
    """,
    
    'PPG Growth': """
**Points Per Game Growth (Year 1 → Year 2)**

**Calculation:** Year 2 PPG minus Year 1 PPG.

**Basketball Context:** Perhaps the most important feature. A positive delta indicates a 
'Sophomore Jump'—showing the player is actively expanding their offensive role and improving. 
Players with +3 PPG growth have 2.1x higher breakout probability.
    """
}

# Add this NEW function
def create_better_trajectory_chart(player_row, pred_p1, pred_p2):
    """
    Creates an improved career trajectory chart with better aesthetics and clarity.
    Shows Year 1, Year 2 (actual), and Year 5 (both predictions + actual if available).
    """
    fig = go.Figure()
    
    # Actual career data (solid line)
    actual_years = [1, 2]
    actual_ppg = [player_row['y1_pts'], player_row['y2_pts']]
    
    fig.add_trace(go.Scatter(
        x=actual_years,
        y=actual_ppg,
        mode='lines+markers',
        name='Actual Career',
        line=dict(color='#2E86AB', width=4),
        marker=dict(size=12, symbol='circle')
    ))
    
    # Path 1 Prediction (from Year 2 to Year 5)
    fig.add_trace(go.Scatter(
        x=[2, 5],
        y=[player_row['y2_pts'], pred_p1],
        mode='lines+markers',
        name='Path 1 Prediction',
        line=dict(color='#A23B72', width=3, dash='dash'),
        marker=dict(size=14, symbol='diamond')
    ))
    
    # Path 2 Prediction (from Year 2 to Year 5)
    fig.add_trace(go.Scatter(
        x=[2, 5],
        y=[player_row['y2_pts'], pred_p2],
        mode='lines+markers',
        name='Path 2 Prediction',
        line=dict(color='#F18F01', width=3, dash='dot'),
        marker=dict(size=14, symbol='diamond')
    ))
    
    # If actual Year 5 exists, add it
    if not pd.isna(player_row['y5_pts']):
        fig.add_trace(go.Scatter(
            x=[5],
            y=[player_row['y5_pts']],
            mode='markers',
            name='Actual Year 5',
            marker=dict(size=16, symbol='star', color='#06A77D', line=dict(width=2, color='white'))
        ))
    
    # Add shaded region showing prediction uncertainty
    fig.add_trace(go.Scatter(
        x=[2, 5, 5, 2],
        y=[player_row['y2_pts'], pred_p1, pred_p2, player_row['y2_pts']],
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.1)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Styling
    fig.update_layout(
        title=f"Career Projection: Years 1-5",
        xaxis_title="NBA Season",
        yaxis_title="Points Per Game",
        xaxis=dict(
            tickmode='array',
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)',
            zeroline=True
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

@st.cache_data
def load_metadata():
    """Load JSON metadata for both model paths."""
    try:
        with open(PATH_COMPARISON_JSON, 'r') as f:
            comparison = json.load(f)
        with open(METADATA_P1_JSON, 'r') as f:
            p1 = json.load(f)
        with open(METADATA_P2_JSON, 'r') as f:
            p2 = json.load(f)
        return comparison, p1, p2
    except Exception as e:
        # Fallback data in case JSONs aren't generated yet
        return {}, {"MAE": "4.92", "Features": 12}, {"MAE": "4.58", "Features": 17}

def calculate_advanced_features(player_row):
    """Calculates Path 2 engineered features for a single player row."""
    # 1. Skill Diversity Score
    sd_score = (
        (player_row['delta_pts'] > 2).astype(int) +
        (player_row['delta_ast'] > 1).astype(int) +
        ((player_row['y2_fg3_pct'] - player_row['y1_fg3_pct']) > 0.03).astype(int)
    )
    
    # 2. Efficiency Maintenance
    usage_eff = player_row['y2_pts'] / (player_row['y2_fg_pct'] + 0.01)
    
    # 3. Draft Position Gap (Expected PPG vs Actual)
    expected_ppg = 20 - (player_row['draft_pick'] * 0.25)
    draft_gap = player_row['y2_pts'] - expected_ppg
    
    # 4. Minutes Momentum
    min_trajectory = player_row['delta_min'] / (player_row['y1_min'] + 1)
    
    # 5. FT % Improvement
    ft_delta = player_row['y2_ft_pct'] - player_row['y1_ft_pct']
    
    return {
        'skill_diversity': sd_score,
        'usage_to_efficiency': usage_eff,
        'overperform_draft': draft_gap,
        'minutes_trajectory': min_trajectory,
        'ft_pct_improvement': ft_delta
    }

def get_archetype_info(cluster_id):
    return ARCHETYPE_MAP.get(cluster_id, {"name": "Unknown", "desc": "N/A"})

@st.cache_resource
def load_artifacts():
    """Load both models and shared scaler/kmeans."""
    model_p1 = joblib.load(MODEL_PATH_P1)
    model_p2 = joblib.load(MODEL_PATH_P2)
    scaler = joblib.load(SCALER_PATH)
    kmeans = joblib.load(KMEANS_PATH)
    return model_p1, model_p2, scaler, kmeans

@st.cache_data
def load_data():
    """Load the processed predictions CSV."""
    df = pd.read_csv(DATA_PATH)
    # Ensure draft_pick is numeric for plotting
    df['draft_pick'] = pd.to_numeric(df['draft_pick'], errors='coerce').fillna(60)
    return df

def get_player_data(df, player_name):
    """Retrieve all stats for a specific player."""
    player_row = df[df['name'] == player_name]
    if not player_row.empty:
        return player_row.iloc[0]
    return None

def get_archetype_name(cluster_id):
    """Convert numeric cluster ID to human-readable string name."""
    info = ARCHETYPE_MAP.get(cluster_id, {"name": "Unknown Archetype"})
    return info["name"]

def format_prediction_metrics(player_row, model, features):
    """Helper to organize data and generate predictions on the fly if missing."""
    
    # 1. Check if we already have a saved prediction in the CSV
    pred_val = player_row.get('predicted_y5_pts', None)
    
    # 2. If it's missing (NaN or N/A), calculate it NOW using the model
    if pd.isna(pred_val) or pred_val == 0:
        # Prepare the features for this specific player
        # We wrap it in a list [[]] because the model expects a 2D array
        feat_input = pd.DataFrame([player_row[features].fillna(0)])
        pred_val = float(model.predict(feat_input)[0])

    metrics = {
        "Year 2 PPG": round(float(player_row['y2_pts']), 1),
        "Predicted Year 5 PPG": round(pred_val, 1),
        "Improvement (Points)": round(float(player_row['delta_pts']), 1),
        "Archetype": get_archetype_name(player_row['cluster'])
    }
    
    if not pd.isna(player_row['y5_pts']):
        metrics["Actual Year 5 PPG"] = round(float(player_row['y5_pts']), 1)
        
    return metrics


def create_radar_chart(player_row, df):
    # Get the average stats for this player's cluster
    cluster_id = player_row['cluster']
    cluster_avg = df[df['cluster'] == cluster_id][['y2_pts', 'y2_reb', 'y2_ast', 'y2_fg_pct']].mean()
    
    categories = ['Points', 'Rebounds', 'Assists', 'FG%']
    
    fig = go.Figure()

    # Player Trace
    fig.add_trace(go.Scatterpolar(
        r=[player_row['y2_pts'], player_row['y2_reb'], player_row['y2_ast'], player_row['y2_fg_pct']*100],
        theta=categories,
        fill='toself',
        name='Player (Y2)'
    ))
    
    # Cluster Average Trace
    fig.add_trace(go.Scatterpolar(
        r=[cluster_avg['y2_pts'], cluster_avg['y2_reb'], cluster_avg['y2_ast'], cluster_avg['y2_fg_pct']*100],
        theta=categories,
        fill='toself',
        name='Cluster Avg'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
        showlegend=True,
        title=f"Comparison: {player_row['name']} vs. Cluster Average"
    )
    return fig

@st.cache_data
def get_player_image(player_name):
    """
    Finds player_id by name using nba_api and returns official headshot URL.
    Returns a 'Logoman' placeholder if the player isn't found.
    """
    try:
        # Search for player by name (case-insensitive)
        nba_players = players.find_players_by_full_name(player_name)
        
        if nba_players:
            # Take the first match and get their ID
            p_id = nba_players[0]['id']
            # Official NBA CDN URL
            return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{p_id}.png"
        
    except Exception as e:
        print(f"Error fetching image for {player_name}: {e}")
        
    # Fallback to NBA Logoman if player not found or error occurs
    return "https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/logoman.png"