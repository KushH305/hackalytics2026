import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from utils import (
    # Data loaders
    load_artifacts, load_data, load_metadata,
    load_cluster_stats, load_feature_importance, load_test_predictions,
    # Helper functions
    get_player_data, get_archetype_info, get_player_image, calculate_advanced_features,
    # Visualization functions
    create_better_trajectory_chart, create_radar_chart,
    # Constants
    ARCHETYPE_MAP, STAT_GLOSSARY, FEATURES_PATH1
)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="The Arc: NBA Predictor", layout="wide", page_icon="🏀")

# ==========================================
# LOAD DATA (ONCE AT STARTUP)
# ==========================================
model_p1, model_p2 = load_artifacts()
meta_p1, meta_p2 = load_metadata()
df = load_data()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Home", 
    "Scouting Report", 
    "The DNA Explorer", 
    "Model Analysis"
])

footer_html = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: white;
    text-align: center;
    padding: 15px 0; /* Increased padding for a better feel */
    border-top: 1px solid #e6e6e6;
    z-index: 100;
    display: flex;
    justify-content: center;
    align-items: center;
}

.footer p {
    margin: 0; /* Removes default paragraph spacing */
    font-size: 18px; /* Bigger font size */
    color: #1a1a1a; /* Darker shade of black */
    font-family: sans-serif;
}

.footer a {
    text-decoration: none;
    color: #1a1a1a; /* Matching the name color to the text */
    font-weight: bold;
}

.footer a:hover {
    color: #000000; /* Turns pure black when hovering */
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .footer {
        background-color: #0e1117;
        border-top: 1px solid #31333f;
    }
    .footer p, .footer a {
        color: #f0f2f6; /* Off-white for readability in dark mode */
    }
}
</style>

<div class="footer">
    <p>By: <a href="https://kushh-portfolio.vercel.app/" target="_blank">Kush Havinal</a></p>
</div>
"""

# ==========================================
# PAGE: HOME
# ==========================================
if page == "Home":
    st.title("The Arc: NBA Career Predictor")
    st.subheader("Predicting Year 5 Performance from Sophomore Signs")
    st.markdown(footer_html, unsafe_allow_html=True)



    st.markdown("""
    ### Project Overview
    In the NBA, the jump from Year 1 to Year 2 is often the greatest indicator of a player's long-term ceiling. 
    **The Arc** uses Machine Learning to analyze "Sophomore Signs"—improvement deltas, efficiency jumps, and usage rates—to predict 
    how a player will perform in their 5th professional season.

    **Key Metrics:**
    - **Models**: XGBoost and Random Forest Regressors with K-Means Clustering
    - **Dataset:** 460+ Players from 2010-2022 Draft Classes
    - **Features:** Points, Rebounds, Assists, Efficiency Deltas, and Draft Pedigree
    """)
    
    st.divider()
    
    st.markdown("### Our Experimental Design")
    st.markdown("""
    I built **two competing methodologies** to understand what works:
    
    - **Path 1 (Baseline):** Simple features + default training → Fast & interpretable
    - **Path 2 (Advanced):** Engineered features + hyperparameter tuning → Optimized & complex
    """)
    
    # Key metrics dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Path 1: The Baseline")
        with st.container(border=True):
            st.metric("Test MAE", f"{meta_p1.get('test_mae', 'N/A')} PPG")
            st.write("---")
            st.write(f"**Features:** {meta_p1.get('num_features', 12)} Standard Stats")
            st.write(f"**Best Model:** {meta_p1.get('best_model', 'XGBoost')}")
            st.write(f"**Tuning:** None (defaults)")
            st.write(f"**Training Time:** ~{meta_p1.get('xgb_time', 2):.0f}s")
    
    with col2:
        st.markdown("### Path 2: The Advanced")
        with st.container(border=True):
            st.metric("Test MAE", f"{meta_p2.get('test_mae', 'N/A')} PPG", 
                     delta=f"{float(meta_p1.get('test_mae', 5)) - float(meta_p2.get('test_mae', 4.58)):.2f}")
            st.write("---")
            st.write(f"**Features:** {meta_p2.get('num_features', 17)} (+ 5 engineered)")
            st.write(f"**Best Model:** {meta_p2.get('best_model', 'XGBoost (Tuned)')}")
            st.write(f"**Tuning:** GridSearchCV (729 combos)")
            st.write(f"**Training Time:** ~{meta_p2.get('xgb_time', 15):.0f}s")
    
    st.divider()
    
    st.markdown("""
    ### Why "The Arc"?
    NBA Front Offices commit hundreds of millions to players based on "potential." **The Arc** removes the guesswork by using **Machine Learning** to identify the "Sophomore Signal"—the statistical inflection point in Year 2 that predicts Year 5 stardom.
    
    **Built for Decision Makers:**
    - **GMs:** Identify "Buy-Low" candidates whose signals outpace current production
    - **Coaches:** Understand player archetypes for optimal roster construction
    - **Analysts:** Data-driven narratives backed by rigorous experimentation
    """)
    
    st.info("Use the sidebar to explore player predictions, understand our methodology, or dive into results analysis")

# ==========================================
# PAGE: SCOUTING REPORT
# ==========================================
elif page == "Scouting Report":
    st.markdown(footer_html, unsafe_allow_html=True)
    st.title("Player Scouting Report")
    st.write("Search any player to see dual-path predictions and career trajectory analysis")
    
    player_list = sorted(df['name'].unique())
    selected_player = st.selectbox("Select a Player to Analyze", player_list, key="player_search")
    
    if selected_player:
        player_row = get_player_data(df, selected_player)
        
        # Calculate advanced features for Path 2
        adv_feats = calculate_advanced_features(player_row)
        player_row_p2 = player_row.copy()
        for k, v in adv_feats.items():
            player_row_p2[k] = v
        
        FEATURES_P2 = FEATURES_PATH1 + list(adv_feats.keys())
        
        # Make predictions
        with st.spinner("Generating dual-path projections..."):
            pred_p1 = float(model_p1.predict(pd.DataFrame([player_row[FEATURES_PATH1].fillna(0)]))[0])
            pred_p2 = float(model_p2.predict(pd.DataFrame([player_row_p2[FEATURES_P2].fillna(0)]))[0])
        
        # Player header
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            img_url = get_player_image(selected_player)
            st.image(img_url, width=200)
        
        with col_info:
            arch_info = get_archetype_info(player_row['cluster'])
            st.header(selected_player)
            st.subheader(f"Archetype: {arch_info['name']}")
            st.caption(arch_info['desc'])
            st.caption(f"Draft: {int(player_row['draft_year'])} (Pick #{int(player_row['draft_pick'])})")
        
        st.divider()
        
        # Current stats
        st.markdown("### Year 2 Performance (Sophomore Season)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("PPG", f"{player_row['y2_pts']:.1f}", delta=f"{player_row['delta_pts']:+.1f}")
        m2.metric("RPG", f"{player_row['y2_reb']:.1f}")
        m3.metric("APG", f"{player_row['y2_ast']:.1f}")
        m4.metric("FG%", f"{player_row['y2_fg_pct']*100:.1f}%")
        m5.metric("MPG", f"{player_row['y2_min']:.1f}")
        
        st.divider()
        
        # Dual predictions
        st.markdown("### Year 5 Projections")
        
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
            st.warning(f"**Model Disagreement:** {diff:.1f} PPG variance between paths")
        else:
            st.success(f"**Model Agreement:** Only {diff:.1f} PPG variance")
        
        # Validation
        if not pd.isna(player_row['y5_pts']):
            actual_y5 = player_row['y5_pts']
            st.divider()
            st.markdown("### Validation: Actual Year 5 Performance")
            
            v1, v2, v3 = st.columns(3)
            v1.metric("Actual Y5 PPG", f"{actual_y5:.1f}")
            
            err_p1 = abs(actual_y5 - pred_p1)
            err_p2 = abs(actual_y5 - pred_p2)
            
            v2.metric("Path 1 Error", f"{err_p1:.1f} PPG", delta_color="inverse")
            v3.metric("Path 2 Error", f"{err_p2:.1f} PPG", delta_color="inverse")
            
            winner = "Path 1" if err_p1 < err_p2 else "Path 2"
            st.info(f"**Winner:** {winner} was more accurate for {selected_player}")
        
        st.divider()
        
        # Visualizations
        col_arc, col_radar = st.columns(2)
        
        with col_arc:
            st.markdown("### Career Trajectory")
            fig = create_better_trajectory_chart(player_row, pred_p1, pred_p2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col_radar:
            st.markdown("### Player vs Archetype")
            fig_radar = create_radar_chart(player_row, df)
            st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# PAGE: DNA EXPLORER
# ==========================================
elif page == "The DNA Explorer":
    st.markdown(footer_html, unsafe_allow_html=True)
    st.title("Player DNA (Clustering)")
    st.markdown("""
    I used **K-Means Clustering** to group players into 5 distinct archetypes based on 
    Year 2 statistics. These archetypes reflect **actual statistical behavior**, not traditional positions.
    """)
    
    st.divider()
    
    # Archetype key
    st.subheader("The 5 Archetypes")
    
    cols = st.columns(5)
    for i, (cluster_id, info) in enumerate(ARCHETYPE_MAP.items()):
        with cols[i]:
            with st.container(border=True):
                st.markdown(f"**{info['name']}**")
                st.caption(info['desc'])
    
    st.divider()
    
    # Cluster stats table
    st.subheader("Average Archetype Statistics")
    
    try:
        cluster_stats = load_cluster_stats().round(2)
        
        cluster_stats['archetype'] = cluster_stats['cluster_id'].map(
            lambda x: ARCHETYPE_MAP[x]['name']
        )
        
        display_cols = ['archetype', 'count', 'y2_pts', 'y2_reb', 'y2_ast', 'y2_fg3_pct', 'delta_pts']
        cluster_stats_display = cluster_stats[display_cols]
        cluster_stats_display.columns = ['Archetype', 'Count', 'Y2 PPG', 'Y2 RPG', 'Y2 APG', 'Y2 3P%', 'PPG Growth']
        
        styled_df = cluster_stats_display.style.highlight_max(
            axis=0, 
            subset=['Y2 PPG', 'Y2 RPG', 'Y2 APG', 'Y2 3P%', 'PPG Growth'], 
            color="#90EE90"
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("**Green highlights** indicate which archetype leads in each statistical category")
        
    except Exception as e:
        st.error(f"Error loading cluster stats: {e}")
    
    # Stat glossary
    st.divider()
    st.subheader("Stat Definitions")
    
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
    
    # Visual map
    st.subheader("Player Archetype Map (2D Projection)")
    
    try:
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

# ==========================================
# PAGE: MODEL ANALYSIS
# ==========================================
elif page == "Model Analysis":
    st.markdown(footer_html, unsafe_allow_html=True)
    st.title("Model Analysis: Baseline vs Advanced")
    st.markdown("""
    Two modeling **paths** predict Year 5 scoring outcomes for NBA players. This page walks through 
    methodology, performance, and error analysis side-by-side.
    """)
    
    st.divider()
    
    # Methodology overview
    st.markdown("## Methodology Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Path 1: Baseline")
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
        st.write(f"**Model:** {meta_p1.get('best_model', 'XGBoost')}")
        st.write("**Tuning:** None (defaults)")
        st.write(f"**Training time:** ~{meta_p1.get('xgb_time', 2):.0f}s")
    
    with col2:
        st.markdown("### Path 2: Advanced")
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
        st.write(f"**Model:** {meta_p2.get('best_model', 'XGBoost (Tuned)')}")
        st.write("**Tuning:** GridSearchCV (729 combos)")
        st.write(f"**Training time:** ~{meta_p2.get('xgb_time', 15):.0f}s")
    
    st.divider()
    
    # Model performance
    st.markdown("## Model Performance")
    
    st.markdown("### Path 1: Baseline Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", f"{meta_p1.get('test_mae', 'N/A')} PPG")
    m2.metric("Test R²", f"{meta_p1.get('test_r2', 'N/A')}")
    m3.metric("Train MAE", f"{meta_p1.get('train_mae', 'N/A')} PPG")
    m4.metric("Improvement vs Naive", f"+{meta_p1.get('improvement_pct', 'N/A')}%")
    
    st.markdown("### Path 2: Advanced Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test MAE", f"{meta_p2.get('test_mae', 'N/A')} PPG")
    m2.metric("Test R²", f"{meta_p2.get('test_r2', 'N/A')}")
    m3.metric("Train MAE", f"{meta_p2.get('train_mae', 'N/A')} PPG")
    m4.metric("Improvement vs Naive", f"+{meta_p2.get('improvement_pct', 'N/A')}%")
    
    st.divider()
    
    # Summary table
    st.markdown("## Head-to-Head Summary")
    
    comparison_data = {
        'Metric': ['Features', 'Tuning', 'Best Model', 'Train MAE', 'Test MAE', 'Test R²', 'Training Time', 'Improvement vs Naive'],
        'Path 1': [
            f"{meta_p1.get('num_features', 12)}",
            'No',
            meta_p1.get('best_model', 'XGBoost'),
            f"{meta_p1.get('train_mae', 'N/A')} PPG",
            f"{meta_p1.get('test_mae', 'N/A')} PPG",
            f"{meta_p1.get('test_r2', 'N/A')}",
            f"~{meta_p1.get('xgb_time', 2):.0f}s",
            f"+{meta_p1.get('improvement_pct', 'N/A')}%"
        ],
        'Path 2': [
            f"{meta_p2.get('num_features', 17)}",
            'Yes (729 combos)',
            meta_p2.get('best_model', 'XGBoost'),
            f"{meta_p2.get('train_mae', 'N/A')} PPG",
            f"{meta_p2.get('test_mae', 'N/A')} PPG",
            f"{meta_p2.get('test_r2', 'N/A')}",
            f"~{meta_p2.get('xgb_time', 15):.0f}s",
            f"+{meta_p2.get('improvement_pct', 'N/A')}%"
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Feature importance
    st.markdown("## Feature Importance: Side-by-Side")
    
    try:
        imp_p1, imp_p2 = load_feature_importance()
        imp_p1 = imp_p1.sort_values('importance', ascending=False).head(10)
        imp_p2 = imp_p2.sort_values('importance', ascending=False).head(10)
        
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
        
    except Exception as e:
        st.warning(f"Feature importance comparison not available: {e}")
    
    st.divider()
    
    # Error analysis
    st.markdown("## Prediction Error Analysis")
    
    try:
        test_p1, test_p2 = load_test_predictions()
        test_p1 = test_p1.sort_values('error')
        test_p2 = test_p2.sort_values('error')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Path 1 Errors")
            fig1 = px.histogram(test_p1, x='error', nbins=15, 
                               title=f'Avg: {test_p1["error"].mean():.2f} PPG')
            fig1.update_layout(height=300)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("### Path 2 Errors")
            fig2 = px.histogram(test_p2, x='error', nbins=15,
                               title=f'Avg: {test_p2["error"].mean():.2f} PPG')
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.divider()
        st.markdown("### Best and Worst Predictions (Path 2)")
        
        col_best, col_worst = st.columns(2)
        
        with col_best:
            st.markdown("#### Top 5 Best")
            for _, row in test_p2.head(5).iterrows():
                st.write(f"**{row['name']}** - Error: {row['error']:.1f} PPG")
        
        with col_worst:
            st.markdown("#### Top 5 Worst")
            for _, row in test_p2.tail(5).iterrows():
                st.write(f"**{row['name']}** - Error: {row['error']:.1f} PPG")
        
    except Exception as e:
        st.error(f"Error loading test data: {e}")