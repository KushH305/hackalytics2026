import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda():
    if not os.path.exists('data/career_arcs.csv'):
        return

    df = pd.read_csv('data/career_arcs.csv')
    
    # Create a figure with two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: The "Growth Curve" (Correlation)
    sns.regplot(data=df, x='y2_pts', y='y5_pts', ax=ax1, 
                scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    ax1.set_title('Predicting the Future: Year 2 vs Year 5 PPG')
    ax1.set_xlabel('Points Per Game (Year 2)')
    ax1.set_ylabel('Points Per Game (Year 5)')

    # Plot 2: Distribution of Draft Picks
    sns.histplot(df['draft_pick'], bins=30, kde=True, ax=ax2, color='green')
    ax2.set_title('Distribution of Draft Picks in Dataset')
    ax2.set_xlabel('Pick Number')

    plt.tight_layout()
    plt.savefig('data/eda_preview.png')
    print("EDA Preview saved to data/eda_preview.png. Check it out!")

if __name__ == "__main__":
    run_eda()