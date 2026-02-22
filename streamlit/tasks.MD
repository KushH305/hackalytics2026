
Ticket #1: The Core Logic (utils.py)
Priority: High | Est. Time: 1 hour

Build the "engine room" so your app pages don't get messy.

[ ] Load Artifacts: Write a function to load the 3 .pkl files and final_player_predictions.csv using @st.cache_resource.

[ ] The Predictor: Create a function predict_y5(player_name) that pulls a player's Year 2 stats and runs them through the XGBoost model.

[ ] The Labeler: Map those Cluster IDs (0, 1, 2, 3) to the human-readable names we discussed (e.g., "Elite Playmakers").

[ ] Error Handling: Ensure that if a player isn't in your CSV, the app doesn't crash (return a "Player Not Found" message).

Ticket #2: The Layout (Multipage app.py)
Priority: High | Est. Time: 1.5 hours

Structure the user journey.

[ ] Home Page: A sleek landing page with your project title ("The Arc: NBA Career Predictor") and a high-level summary of the model (MAE 1.21).

[ ] Scouting Report (Page 1): A search bar where you select a player and see their Predicted vs. Actual stats and their Archetype.

[ ] The "DNA" Explorer (Page 2): A page showing your Cluster Stats—visualize what makes an "Elite Scorer" vs. a "3-and-D Wing."

[ ] Model Transparency (Page 3): Display the feature_importance.csv so the judges see that you aren't just "guessing."

Ticket #3: Data Visualization (Plotly)
Priority: Medium | Est. Time: 2 hours

Make the data "pop" for the judges.

[ ] Career Arc Line Chart: Plot Year 1, Year 2, and the Predicted Year 5 stats on a single timeline.

[ ] Radar Chart: Create a "Player Profile" radar chart showing REB, AST, PTS, and FG% relative to their cluster average.

[ ] Feature Importance Bar Chart: A simple horizontal bar chart showing that "Delta PTS" or "Minutes" were your top predictors.

Ticket #4: The "Stress Test" (QA)
Priority: High | Est. Time: 1 hour

Make it "Hackathon-Proof."

[ ] The "Five Player" Test: Test a star (Aaron Gordon), a bust, a role player (Aaron Holiday), a rookie (AJ Griffin), and a missing player.

[ ] Null Handling: Check what happens if a player is missing Year 2 stats. (The app should show a "Insufficient Data" warning, not a red error code).

[ ] Final CSV Sync: Ensure the final_player_predictions.csv in your folder matches exactly what the app is reading