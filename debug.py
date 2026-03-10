# debug_model.py
import pandas as pd
import numpy as np
import joblib
import sys
import os

# Get absolute paths - YOUR STRUCTURE HAS updated_models AT ROOT, NOT IN streamlit/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'updated_models')
DATA_DIR = os.path.join(MODELS_DIR, 'data')
MODEL_DIR = os.path.join(MODELS_DIR, 'models')
STREAMLIT_DIR = os.path.join(SCRIPT_DIR, 'streamlit')

# Add streamlit folder to path so we can import utils
sys.path.insert(0, STREAMLIT_DIR)

from utils import calculate_advanced_features, FEATURES_PATH1

print("="*60)
print("MODEL HEALTH CHECK")
print("="*60)
print(f"\nScript location: {SCRIPT_DIR}")
print(f"Models folder: {MODELS_DIR}")
print(f"Data folder: {DATA_DIR}")

# Check if folders exist
if not os.path.exists(DATA_DIR):
    print(f"\n❌ ERROR: Data folder doesn't exist!")
    print(f"   Expected: {DATA_DIR}")
    sys.exit(1)

# Load data
print("\n1. Loading data...")
csv_path = os.path.join(DATA_DIR, 'final_player_predictions.csv')
print(f"   Looking for: {csv_path}")

if not os.path.exists(csv_path):
    print(f"   ❌ File not found!")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df)} players")

# Get Anthony Davis
ad = df[df['name'] == 'Anthony Davis']
if ad.empty:
    print("\n❌ Anthony Davis not found in dataset!")
    print(f"   Available players (first 10): {list(df['name'].head(10))}")
    sys.exit(1)

ad = ad.iloc[0]
print(f"\n2. Found Anthony Davis (draft year: {ad['draft_year']:.0f})")

# Load models
print("\n3. Loading models...")
model_p1_path = os.path.join(MODEL_DIR, 'path1', 'final_model.pkl')
model_p2_path = os.path.join(MODEL_DIR, 'path2', 'final_model.pkl')

print(f"   Path 1: {model_p1_path}")
if not os.path.exists(model_p1_path):
    print(f"   ❌ Path 1 model not found!")
    sys.exit(1)

print(f"   Path 2: {model_p2_path}")
if not os.path.exists(model_p2_path):
    print(f"   ❌ Path 2 model not found!")
    sys.exit(1)

model_p1 = joblib.load(model_p1_path)
model_p2 = joblib.load(model_p2_path)
print(f"✓ Path 1 model type: {type(model_p1).__name__}")
print(f"✓ Path 2 model type: {type(model_p2).__name__}")

# Check model parameters
print("\n4. Model parameters:")
print(f"Path 1 params: {model_p1.get_params()}")
print(f"\nPath 2 params: {model_p2.get_params()}")

# Check if models have feature names
print("\n5. Checking expected features...")
if hasattr(model_p1, 'feature_names_in_'):
    print(f"Path 1 expects {len(model_p1.feature_names_in_)} features:")
    print(model_p1.feature_names_in_)
else:
    print("Path 1: No feature_names_in_ attribute (older scikit-learn)")

if hasattr(model_p2, 'feature_names_in_'):
    print(f"\nPath 2 expects {len(model_p2.feature_names_in_)} features:")
    print(model_p2.feature_names_in_)
else:
    print("Path 2: No feature_names_in_ attribute")

# Print AD's Year 2 stats
print("\n6. Anthony Davis Year 2 stats:")
print(f"   Y2 PPG: {ad['y2_pts']:.1f}")
print(f"   Y2 RPG: {ad['y2_reb']:.1f}")
print(f"   Y2 APG: {ad['y2_ast']:.1f}")
print(f"   Y2 FG%: {ad['y2_fg_pct']:.3f}")
print(f"   Draft Pick: {ad['draft_pick']:.0f}")

# Calculate advanced features
print("\n7. Calculating advanced features...")
adv_feats = calculate_advanced_features(ad)
print("Advanced features calculated:")
for k, v in adv_feats.items():
    print(f"   {k}: {v:.4f} (type: {type(v).__name__})")

# Prepare Path 1 input
print("\n8. Preparing Path 1 input...")
X_p1 = pd.DataFrame([ad[FEATURES_PATH1].fillna(0)])
print(f"Input shape: {X_p1.shape}")
print(f"Features: {list(X_p1.columns)}")
print("\nFirst 5 values:")
for col in list(X_p1.columns)[:5]:
    print(f"   {col}: {X_p1[col].values[0]:.4f}")

# Path 1 Prediction
print("\n9. PATH 1 PREDICTION:")
pred_p1 = model_p1.predict(X_p1)[0]
print(f"   Result: {pred_p1:.2f} PPG")

# Prepare Path 2 input
print("\n10. Preparing Path 2 input...")
ad_p2 = ad.copy()
for k, v in adv_feats.items():
    ad_p2[k] = v

FEATURES_P2 = FEATURES_PATH1 + list(adv_feats.keys())
X_p2 = pd.DataFrame([ad_p2[FEATURES_P2].fillna(0)])
print(f"Input shape: {X_p2.shape}")
print(f"Features: {list(X_p2.columns)}")
print("\nLast 5 values (advanced features):")
for col in list(X_p2.columns)[-5:]:
    print(f"   {col}: {X_p2[col].values[0]:.4f}")

# Path 2 Prediction
print("\n11. PATH 2 PREDICTION:")
pred_p2 = model_p2.predict(X_p2)[0]
print(f"   Result: {pred_p2:.2f} PPG")

# Compare to stored predictions
print("\n12. Comparing to stored predictions in CSV:")
if 'predicted_y5_pts_path1' in ad.index:
    stored_p1 = ad['predicted_y5_pts_path1']
    print(f"   Stored Path 1: {stored_p1:.2f} PPG")
    print(f"   Fresh Path 1:  {pred_p1:.2f} PPG")
    print(f"   Difference: {abs(stored_p1 - pred_p1):.4f}")
else:
    print("   No stored Path 1 prediction in CSV")

if 'predicted_y5_pts_path2' in ad.index:
    stored_p2 = ad['predicted_y5_pts_path2']
    print(f"\n   Stored Path 2: {stored_p2:.2f} PPG")
    print(f"   Fresh Path 2:  {pred_p2:.2f} PPG")
    print(f"   Difference: {abs(stored_p2 - pred_p2):.4f}")
else:
    print("   No stored Path 2 prediction in CSV")

# Actual Year 5
if not pd.isna(ad['y5_pts']):
    print(f"\n13. Actual Year 5: {ad['y5_pts']:.2f} PPG")
    print(f"   Path 1 error: {abs(ad['y5_pts'] - pred_p1):.2f} PPG")
    print(f"   Path 2 error: {abs(ad['y5_pts'] - pred_p2):.2f} PPG")

# Test on a few more players
print("\n14. Testing on other players:")
test_players = ['LeBron James', 'Stephen Curry', 'Giannis Antetokounmpo']

for player_name in test_players:
    player_data = df[df['name'] == player_name]
    if player_data.empty:
        print(f"   {player_name}: NOT FOUND")
        continue
    
    p = player_data.iloc[0]
    
    # Path 1
    X_p1_test = pd.DataFrame([p[FEATURES_PATH1].fillna(0)])
    pred_p1_test = model_p1.predict(X_p1_test)[0]
    
    # Path 2
    adv_test = calculate_advanced_features(p)
    p_p2 = p.copy()
    for k, v in adv_test.items():
        p_p2[k] = v
    X_p2_test = pd.DataFrame([p_p2[FEATURES_P2].fillna(0)])
    pred_p2_test = model_p2.predict(X_p2_test)[0]
    
    print(f"   {player_name}:")
    print(f"      Path 1: {pred_p1_test:.1f} | Path 2: {pred_p2_test:.1f}")

print("\n" + "="*60)
print("DEBUG COMPLETE")
print("="*60)