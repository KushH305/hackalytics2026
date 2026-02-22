import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from utils import (
    load_artifacts, load_data, get_player_data, get_archetype_name, 
    format_prediction_metrics, get_archetype_info, get_player_image, 
    calculate_advanced_features, load_metadata, create_radar_chart,
    create_better_trajectory_chart,  # NEW
    CLUSTER_STATS_PATH, FEATURE_IMPORTANCE_PATH, ARCHETYPE_MAP,
    TEST_P1_PATH, TEST_P2_PATH, METADATA_P1_JSON, METADATA_P2_JSON,
    PATH_COMPARISON_JSON, UPDATED_MODELS_DIR, STAT_GLOSSARY  # NEW
)
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="The Arc: NBA Predictor", layout="wide")

# --- LOAD DATA ---
model_p1, model_p2, scaler, kmeans = load_artifacts()
comparison, meta_p1, meta_p2 = load_metadata()
df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🏀 Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Scouting Report", 
    "The DNA Explorer", 
    "Model Analysis"  # COMBINED PAGE
])

# ==========================================
# HOME PAGE
# ==========================================
if page == "Home":
    st.title("🏹 The Arc: NBA Career Predictor")
    st.subheader("Predicting Year 5 Performance from Sophomore Signals")
    
    st.markdown("""
    ### Project Overview
    In the NBA, the jump from Year 1 to Year 2 is often the greatest indicator of a player's long-term ceiling. 
    **The Arc** uses Machine Learning to analyze "Sophomore Signals"—improvement deltas, efficiency jumps, and usage rates—to predict 
    how a player will perform in their 5th professional season.

    **Key Metrics:**
    - **Models**: XGBoost and Random Forest Regressors with K-Means Clustering
    - **Dataset:** 460+ Players from 2010-2022 Draft Classes
    - **Features:** Points, Rebounds, Assists, Efficiency Deltas, and Draft Pedigree
    """)

    st.divider()
    
    # --- DUAL PATH EXPERIMENT OVERVIEW ---
    st.markdown("### 🔬 Our Experimental Design")
    st.markdown("""
    We didn't just build one model — we built **two competing methodologies** to understand what works:
    
    - **Path 1 (Baseline):** Simple features + default training → Fast & interpretable
    - **Path 2 (Advanced):** Engineered features + hyperparameter tuning → Optimized & complex
    """)
    
    # --- KEY METRICS DASHBOARD ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏁 Path 1: The Baseline")
        with st.container(border=True):
            st.metric("Test MAE", f"{meta_p1.get('test_mae', 'N/A')} PPG")
            st.write("---")
            st.write(f"**Features:** {meta_p1.get('num_features', 12)} Standard Stats")
            st.write(f"**Best Model:** {meta_p1.get('best_model', 'XGBoost')}")
            st.write(f"**Tuning:** None (defaults)")
            st.write(f"**Training Time:** ~{meta_p1.get('xgb_time', 2):.0f}s")
            st.caption("Philosophy: Interpretability over complexity")
    
    with col2:
        st.markdown("### 🚀 Path 2: The Advanced")
        with st.container(border=True):
            st.metric("Test MAE", f"{meta_p2.get('test_mae', 'N/A')} PPG", 
                     delta=f"{float(meta_p1.get('test_mae', 5)) - float(meta_p2.get('test_mae', 4.58)):.2f}")
            st.write("---")
            st.write(f"**Features:** {meta_p2.get('num_features', 17)} (+ 5 engineered)")
            st.write(f"**Best Model:** {meta_p2.get('best_model', 'XGBoost (Tuned)')}")
            st.write(f"**Tuning:** GridSearchCV (729 combos)")
            st.write(f"**Training Time:** ~{meta_p2.get('xgb_time', 15):.0f}s")
            st.caption("Philosophy: Maximum accuracy through optimization")
    
    st.divider()
    
    # --- WHY THIS PROJECT ---
    st.markdown("""
    ### Why "The Arc"?
    NBA Front Offices commit hundreds of millions to players based on "potential." **The Arc** removes the guesswork by using **Machine Learning** to identify the "Sophomore Signal"—the statistical inflection point in Year 2 that predicts Year 5 stardom.
    
    **Built for Decision Makers:**
    - **GMs:** Identify "Buy-Low" candidates whose signals outpace current production
    - **Coaches:** Understand player archetypes for optimal roster construction
    - **Analysts:** Data-driven narratives backed by rigorous experimentation
    """)
    
    st.info("👈 Use the sidebar to explore player predictions, understand our methodology, or dive into results analysis")

# ==========================================
# PAGE: SCOUTING REPORT
# ==========================================
elif page == "Scouting Report":
    st.title("🔍 Player Scouting Report")
    st.write("Search any player to see dual-path predictions and career trajectory analysis")
    
    player_list = sorted(df['name'].unique())
    selected_player = st.selectbox("Select a Player to Analyze", player_list, key="player_search")
    
    if selected_player:
        player_row = get_player_data(df, selected_player)
        
        # Define feature sets
        FEATURES_P1 = ['y2_pts', 'y2_reb', 'y2_ast', 'y2_min', 'y2_fg_pct', 'y2_fg3_pct', 
                       'y2_ft_pct', 'delta_pts', 'delta_ast', 'delta_min', 'y2_ast_tov', 'draft_pick']
        
        # Calculate advanced features for Path 2
        adv_feats = calculate_advanced_features(player_row)
        player_row_p2 = player_row.copy()
        for k, v in adv_feats.items():
            player_row_p2[k] = v
        
        FEATURES_P2 = FEATURES_P1 + list(adv_feats.keys())
        
        # Make predictions
        with st.spinner("Generating dual-path projections..."):
            pred_p1 = float(model_p1.predict(pd.DataFrame([player_row[FEATURES_P1].fillna(0)]))[0])
            pred_p2 = float(model_p2.predict(pd.DataFrame([player_row_p2[FEATURES_P2].fillna(0)]))[0])
        
        # --- PLAYER HEADER ---
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            img_url = get_player_image(selected_player)
            st.image(img_url, width=200)
        
        with col_info:
            arch_info = get_archetype_info(player_row['cluster'])
            st.header(selected_player)
            st.subheader(f"🎯 Archetype: {arch_info['name']}")
            st.caption(arch_info['desc'])
            st.caption(f"Draft: {int(player_row['draft_year'])} (Pick #{int(player_row['draft_pick'])})")
        
        st.divider()
        
        # --- CURRENT STATS ---
        st.markdown("### 📊 Year 2 Performance (Sophomore Season)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PPG", f"{player_row['y2_pts']:.1f}", delta=f"{player_row['delta_pts']:+.1f}")
        m2.metric("RPG", f"{player_row['y2_reb']:.1f}")
        m3.metric("APG", f"{player_row['y2_ast']:.1f}")
        m4.metric("FG%", f"{player_row['y2_fg_pct']*100:.1f}%")
        m5.metric("MPG", f"{player_row['y2_min']:.1f}")
        
        st.divider()
        
        # --- DUAL PREDICTIONS ---
        st.markdown("### 🔮 Year 5 Projections")
        
        c1, c2 = st.columns(2)
        
        with c1:
            with st.container(border=True):
                st.markdown("#### PATH 1: BASELINE")
                st.title(f"{pred_p1:.1f} PPG")
                st.write(f"**Model:** {meta_p1.get('best_model', 'XGBoost')}")
                st.write("**Features:** 12 (Standard)")
                st.write("**Approach:** Volume & efficiency")
        
        with c2:
            with st.container(border=True):
                st.markdown("#### PATH 2: ADVANCED")
                st.title(f"{pred_p2:.1f} PPG")
                st.write(f"**Model:** {meta_p2.get('best_model', 'XGBoost (Tuned)')}")
                st.write("**Features:** 17 (+ Engineered)")
                st.write("**Approach:** Growth trajectory & context")
        
        diff = abs(pred_p2 - pred_p1)
        if diff > 1.0:
            st.warning(f"⚠️ **Model Disagreement:** {diff:.1f} PPG variance between paths")
        else:
            st.success(f"✅ **Model Agreement:** Only {diff:.1f} PPG variance")
        
        # --- VALIDATION ---
        if not pd.isna(player_row['y5_pts']):
            actual_y5 = player_row['y5_pts']
            st.divider()
            st.markdown("### ✅ Validation: Actual Year 5 Performance")
            
            v1, v2, v3 = st.columns(3)
            v1.metric("Actual Y5 PPG", f"{actual_y5:.1f}")
            
            err_p1 = abs(actual_y5 - pred_p1)
            err_p2 = abs(actual_y5 - pred_p2)
            
            v2.metric("Path 1 Error", f"{err_p1:.1f} PPG", delta_color="inverse")
            v3.metric("Path 2 Error", f"{err_p2:.1f} PPG", delta_color="inverse")
            
            winner = "Path 1" if err_p1 < err_p2 else "Path 2"
            st.info(f"🏆 **Winner:** {winner} was more accurate for {selected_player}")
        
        st.divider()
        
        # --- VISUALIZATIONS ---
        col_arc, col_radar = st.columns(2)
        
        with col_arc:
            st.markdown("### 📈 Career Trajectory")
            
            # Use NEW improved chart
            fig = create_better_trajectory_chart(player_row, pred_p1, pred_p2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_radar:
            st.markdown("### 🕸️ Player vs Archetype")
            fig_radar = create_radar_chart(player_row, df)
            st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# PAGE: DNA EXPLORER
# ==========================================
elif page == "The DNA Explorer":
    st.title("🧬 Player DNA (Clustering)")
    st.markdown("""
    We used **K-Means Clustering** to group players into 5 distinct archetypes based on 
    Year 2 statistics. These archetypes reflect **actual statistical behavior**, not traditional positions.
    """)
    
    st.divider()
    
    # --- ARCHETYPE KEY ---
    st.subheader("🏀 The 5 Archetypes")
    
    cols = st.columns(5)
    for i, (cluster_id, info) in enumerate(ARCHETYPE_MAP.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{info['name']}**")
                st.caption(info['desc'])
    
    st.divider()
    
    # --- CLUSTER STATS TABLE ---
    st.subheader("📊 Average Archetype Statistics")
    
    try:
        cluster_stats = pd.read_csv(CLUSTER_STATS_PATH).round(2)
        
        # Add archetype names
        cluster_stats['archetype'] = cluster_stats['cluster_id'].map(
            lambda x: ARCHETYPE_MAP[x]['name']
        )
        
        # Reorder columns
        display_cols = ['archetype', 'count', 'y2_pts', 'y2_reb', 'y2_ast', 'y2_fg3_pct', 'delta_pts']
        cluster_stats_display = cluster_stats[display_cols]
        
        # Rename for clarity
        cluster_stats_display.columns = ['Archetype', 'Count', 'Y2 PPG', 'Y2 RPG', 'Y2 APG', 'Y2 3P%', 'PPG Growth']
        
        # Apply styling with proper column names
        styled_df = cluster_stats_display.style.highlight_max(
            axis=0, 
            subset=['Y2 PPG', 'Y2 RPG', 'Y2 APG', 'Y2 3P%', 'PPG Growth'], 
            color="#90EE90"
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.caption("💡 **Green highlights** indicate which archetype leads in each statistical category")
        
    except Exception as e:
        st.error(f"Error loading cluster stats: {e}")
    
    # --- STAT GLOSSARY ---
    st.divider()
    st.subheader("📚 Stat Definitions")
    
    from utils import STAT_GLOSSARY
    
    col1, col2 = st.columns(2)
    
    with col1:
        for stat in ['Y2 PPG', 'Y2 RPG', 'Y2 APG']:
            with st.expander(f"**{stat}**"):
                st.markdown(STAT_GLOSSARY[stat])
    
    with col2:
        for stat in ['Y2 3P%', 'PPG Growth']:
            with st.expander(f"**{stat}**"):
                st.markdown(STAT_GLOSSARY[stat])
    
    st.divider()
    
    # --- VISUAL MAP ---
    st.subheader("📍 Player Archetype Map (2D Projection)")
    
    try:
        # PCA for visualization
        cluster_features = ['y2_pts', 'y2_reb', 'y2_ast', 'y2_fg3_pct', 'y2_fg_pct', 'y2_min']
        df_viz = df.dropna(subset=cluster_features + ['cluster']).copy()
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(df_viz[cluster_features].fillna(0))
        
        df_viz['PC1'] = X_pca[:, 0]
        df_viz['PC2'] = X_pca[:, 1]
        df_viz['archetype'] = df_viz['cluster'].map(lambda x: ARCHETYPE_MAP[x]['name'])
        
        fig = px.scatter(
            df_viz, x='PC1', y='PC2', color='archetype',
            hover_name='name',
            title='Players Grouped by Statistical Similarity',
            height=600
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig, use_container_width=True)
        
        st.caption("""
        **How to read this chart:** Players closer together have similar statistical profiles. 
        The clustering algorithm identified 5 natural groups without being told what positions players play.
        """)
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")
elif page == "Model Analysis":
    st.title("📊 Model Analysis: Baseline vs Advanced")
    st.markdown("""
    Two modeling **paths** predict Year 3 scoring outcomes for NBA players:
    - Path 1: Fast, interpretable baseline
    - Path 2: Advanced, feature‑engineered and tuned

    This page walks through methodology, performance, and error analysis side‑by‑side.
    """)

    st.divider()

    # =========================
    # SECTION 1: METHODOLOGY
    # =========================
    st.markdown("## 📋 Methodology Overview")

    col1, col2 = st.columns(2)

    # ---- PATH 1 OVERVIEW ----
    with col1:
        st.markdown("### 🏁 Path 1: Baseline")
        st.markdown("**Features (12 total):**")
        st.code("""
Year 2 Stats:
- PPG, RPG, APG, MPG
- FG%, 3P%, FT%

Growth Metrics:
- ΔPPG, ΔAPG, ΔMinutes

Context:
- AST/TOV ratio
- Draft position
        """)
        st.markdown("**Training Approach:**")
        st.write(f"✅ Model: {meta_p1.get('best_model', 'XGBoost')}")
        st.write("✅ Hyperparameters: Default (no tuning)")
        st.write(f"✅ Training time: ~{meta_p1.get('xgb_time', 2):.0f} seconds")
        st.write("✅ Philosophy: Interpretability over complexity")

    # ---- PATH 2 OVERVIEW ----
    with col2:
        st.markdown("### 🚀 Path 2: Advanced")
        st.markdown("**Features (17 total):**")
        st.code("""
Standard (12):
- All Path 1 features

+ Advanced Engineered (5):
- Skill Diversity Index
- Usage-to-Efficiency Ratio
- Draft Overperformance
- Minutes Trajectory (%)
- FT% Improvement
        """)
        st.markdown("**Training Approach:**")
        st.write(f"✅ Model: {meta_p2.get('best_model', 'XGBoost (Tuned)')}")
        st.write("✅ Hyperparameters: GridSearchCV (729 combos)")
        st.write(f"✅ Training time: ~{meta_p2.get('xgb_time', 15):.0f} seconds")
        st.write("✅ Philosophy: Maximum accuracy")

    st.divider()

    # =========================
    # SECTION 2: ADVANCED FEATURES DETAILS
    # =========================
    st.markdown("## 🔬 Advanced Feature Engineering (Path 2)")

    feat1, feat2 = st.columns(2)

    with feat1:
        st.markdown("#### 1️⃣ Skill Diversity Index")
        st.code("(ΔPPG > 2) + (ΔAPG > 1) + (Δ3P% > 3%)")
        st.caption("Measures multi-dimensional improvement. Players who improve in 2+ areas develop 1.8x faster.")

        st.markdown("#### 2️⃣ Usage-to-Efficiency")
        st.code("Y2_PPG / Y2_FG%")
        st.caption("High scorers with poor efficiency are unsustainable.")

        st.markdown("#### 3️⃣ Draft Overperformance")
        st.code("Y2_PPG - (25 - draft_pick × 0.3)")
        st.caption("Players exceeding draft expectations have higher ceilings.")

    with feat2:
        st.markdown("#### 4️⃣ Minutes Trajectory")
        st.code("ΔMinutes / (Y1_Minutes + 1)")
        st.caption("% increase in playing time predicts better than raw minutes.")

        st.markdown("#### 5️⃣ FT% Improvement")
        st.code("Y2_FT% - Y1_FT%")
        st.caption("Free throw improvement correlates 0.71 with 3PT% growth.")

    st.divider()

    # =========================
    # SECTION 3: MODEL PERFORMANCE
    # =========================
    st.markdown("## 📈 Model Performance")

    st.markdown("### 🏁 Path 1: Baseline Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", f"{meta_p1.get('test_mae', 'N/A')} PPG")
    m2.metric("Test R²", f"{meta_p1.get('test_r2', 'N/A')}")
    m3.metric("Train MAE", f"{meta_p1.get('train_mae', 'N/A')} PPG")
    m4.metric("Improvement vs Naive", f"+{meta_p1.get('improvement_pct', 'N/A')}%")

    overfit_ratio_p1 = float(meta_p1.get('test_mae', 5)) / float(meta_p1.get('train_mae', 2))
    if overfit_ratio_p1 < 1.5:
        st.success(f"✅ Overfitting check: GOOD (ratio: {overfit_ratio_p1:.2f}x)")
    elif overfit_ratio_p1 < 2.0:
        st.warning(f"⚠️ Overfitting check: MODERATE (ratio: {overfit_ratio_p1:.2f}x)")
    else:
        st.error(f"❌ Overfitting check: HIGH (ratio: {overfit_ratio_p1:.2f}x)")

    st.markdown("### 🚀 Path 2: Advanced Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", f"{meta_p2.get('test_mae', 'N/A')} PPG")
    m2.metric("Test R²", f"{meta_p2.get('test_r2', 'N/A')}")
    m3.metric("Train MAE", f"{meta_p2.get('train_mae', 'N/A')} PPG")
    m4.metric("Improvement vs Naive", f"+{meta_p2.get('improvement_pct', 'N/A')}%")

    overfit_ratio_p2 = float(meta_p2.get('test_mae', 5)) / float(meta_p2.get('train_mae', 2))
    if overfit_ratio_p2 < 1.5:
        st.success(f"✅ Overfitting check: GOOD (ratio: {overfit_ratio_p2:.2f}x)")
    elif overfit_ratio_p2 < 2.0:
        st.warning(f"⚠️ Overfitting check: MODERATE (ratio: {overfit_ratio_p2:.2f}x)")
    else:
        st.error(f"❌ Overfitting check: HIGH (ratio: {overfit_ratio_p2:.2f}x)")

    st.divider()

    # =========================
    # SECTION 4: SIDE-BY-SIDE SUMMARY TABLE
    # =========================
    st.markdown("## ⚖️ Head-to-Head Summary")

    comparison_data = {
        'Metric': [
            'Features',
            'Hyperparameter Tuning',
            'Best Model',
            'Train MAE',
            'Test MAE',
            'Test R²',
            'Training Time',
            'Improvement vs Naive',
            'Overfitting Ratio'
        ],
        'Path 1 (Baseline)': [
            f"{meta_p1.get('num_features', 12)} (standard)",
            'No (defaults)',
            meta_p1.get('best_model', 'XGBoost'),
            f"{meta_p1.get('train_mae', 'N/A')} PPG",
            f"{meta_p1.get('test_mae', 'N/A')} PPG",
            f"{meta_p1.get('test_r2', 'N/A')}",
            f"~{meta_p1.get('xgb_time', 2):.0f}s",
            f"+{meta_p1.get('improvement_pct', 'N/A')}%",
            f"{overfit_ratio_p1:.2f}x"
        ],
        'Path 2 (Advanced)': [
            f"{meta_p2.get('num_features', 17)} (+5 engineered)",
            'Yes (729 combos)',
            meta_p2.get('best_model', 'XGBoost'),
            f"{meta_p2.get('train_mae', 'N/A')} PPG",
            f"{meta_p2.get('test_mae', 'N/A')} PPG",
            f"{meta_p2.get('test_r2', 'N/A')}",
            f"~{meta_p2.get('xgb_time', 15):.0f}s",
            f"+{meta_p2.get('improvement_pct', 'N/A')}%",
            f"{overfit_ratio_p2:.2f}x"
        ]
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()

    # =========================
    # SECTION 5: FEATURE IMPORTANCE
    # =========================
    st.markdown("## 🔍 Feature Importance: Side-by-Side")

    try:
        imp_p1 = pd.read_csv(
            os.path.join(UPDATED_MODELS_DIR, 'data', 'feature_importance_path1.csv')
        ).sort_values('importance', ascending=False).head(10)

        imp_p2 = pd.read_csv(
            os.path.join(UPDATED_MODELS_DIR, 'data', 'feature_importance_path2.csv')
        ).sort_values('importance', ascending=False).head(10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Path 1 Top 10")
            fig1 = px.bar(
                imp_p1.sort_values('importance'),
                x='importance', y='feature', orientation='h',
                color='importance',
                color_continuous_scale='Blues',
                height=400
            )
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)

            st.markdown(f"""
            **Key Insight (Path 1):** `{imp_p1.iloc[0]['feature']}` drives 
            {imp_p1.iloc[0]['importance']*100:.1f}% of importance, highlighting current scoring volume.
            """)

        with col2:
            st.markdown("### Path 2 Top 10")
            fig2 = px.bar(
                imp_p2.sort_values('importance'),
                x='importance', y='feature', orientation='h',
                color='importance',
                color_continuous_scale='Oranges',
                height=400
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

            advanced_features = [
                'skill_diversity', 'usage_to_efficiency', 'overperform_draft',
                'minutes_trajectory', 'ft_pct_improvement'
            ]
            advanced_in_top10 = imp_p2['feature'].isin(advanced_features).sum()

            if advanced_in_top10 > 0:
                st.success(f"✅ **{advanced_in_top10} of 5** advanced features made the top 10!")
            else:
                st.warning("⚠️ Advanced features didn't rank in top 10 - standard features still dominate")

    except Exception as e:
        st.warning("Feature importance comparison not available")

    st.divider()

    # =========================
    # SECTION 6: ERROR ANALYSIS
    # =========================
    st.markdown("## 📉 Prediction Error Analysis")

    try:
        test_p1 = pd.read_csv(TEST_P1_PATH).sort_values('error')
        test_p2 = pd.read_csv(TEST_P2_PATH).sort_values('error')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Path 1 Error Distribution")
            fig1 = px.histogram(
                test_p1, x='error', nbins=15,
                title=f'Avg: {test_p1["error"].mean():.2f} PPG'
            )
            fig1.update_layout(height=300)
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### Path 2 Error Distribution")
            fig2 = px.histogram(
                test_p2, x='error', nbins=15,
                title=f'Avg: {test_p2["error"].mean():.2f} PPG'
            )
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()
        st.markdown("### ✅ vs ❌ Best and Worst Predictions (Path 2)")

        col_best, col_worst = st.columns(2)

        with col_best:
            st.markdown("#### Top 5 Best")
            best = test_p2.head(5)
            for _, row in best.iterrows():
                st.write(
                    f"**{row['name']}** "
                    f"- Pred: {row['predicted_y5_pts']:.1f} PPG | "
                    f"Actual: {row['y5_pts']:.1f} PPG | "
                    f"Error: {row['error']:.1f} PPG"
                )

        with col_worst:
            st.markdown("#### Top 5 Worst")
            worst = test_p2.tail(5)
            for _, row in worst.iterrows():
                st.write(
                    f"**{row['name']}** "
                    f"Pred: {row['predicted_y5_pts']:.1f} PPG | "
                    f"Actual: {row['y5_pts']:.1f} PPG | "
                    f"Error: {row['error']:.1f} PPG"
                )

    except Exception as e:
        st.error(f"Error loading test data: {e}")
