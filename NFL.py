import pandas as pd
import os
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample

current_week = 9

data_folder = '2025NFLData'
#filepath = os.path.join(data_folder, 'Week2_Training_Data.csv')
#filepath2 = os.path.join(data_folder, 'Week3_Prediction.csv')

training_files = [
    f'Week{i}_Training_Data.csv' for i in range(2, current_week)
]

pred_file = f'Week{current_week}_Prediction.csv'

training_dfs = []
for file in training_files:
    path = os.path.join(data_folder, file)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f'Loaded training data: {file}')
        training_dfs.append(df)
    else:
        print(f'Error: Missing training file {file}')

if not training_dfs:
    raise FileNotFoundError('No Training Data Found')

full_train_df = pd.concat(training_dfs, ignore_index=True)

# Week 2 training csv - training using WK2 matchups/outcomes with WK1 data
#df_wk2_train = pd.read_csv('2025NFLData/Week2_Training_Data.csv')
# Week 3 prediction csv - predicting WK3 matchup outcomes using WK1 + WK2 data 
#df_wk3_pred = pd.read_csv('2025NFLData/Week3_Prediction.csv')
# Week 3 training csv - training using WK3 matchups/outcomes with WK1 + WK2 data
#df_wk3_train = pd.read_csv('2025NFLData/Week3_Training_Data.csv')
# Week 4 prediction csv - predicting WK4 matchup outcomes using WK1 - WK3 data 
#df_wk4_pred = pd.read_csv('2025NFLData/Week4_Prediction.csv')

#training_data = pd.concat([df_wk2_train, df_wk3_train], ignore_index=True)

target = 'Outcome'
features = [col for col in full_train_df.columns if col not in ['Home', 'Away', 'Outcome']]

X = full_train_df[features]
y = full_train_df[target]

df_combined = pd.concat([X, y], axis=1)

# Separate by outcome
home_wins = df_combined[df_combined['Outcome'] == 1]  # Home Win
away_wins = df_combined[df_combined['Outcome'] == 0]  # Away Win

# Downsample majority class
n_samples = min(len(home_wins), len(away_wins))

home_balanced = resample(home_wins, n_samples=n_samples, random_state=42)
away_balanced = resample(away_wins, n_samples=n_samples, random_state=42)

# Recombine into balanced dataset
balanced_df = pd.concat([home_balanced, away_balanced], ignore_index=True)
X = balanced_df[features]  # Overwrite X
y = balanced_df['Outcome']  # Overwrite y

print(f"Balanced training set: {len(balanced_df)} games ({n_samples} home wins, {n_samples} away wins)")

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# Evaluate performance using CV
cv = StratifiedKFold(n_splits=min(4, len(y)), shuffle=True, random_state=42)
if len(y) >= 4:
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring='accuracy')
    print(f"\nModel Accuracy (Cross-Validated): {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Retrain on ALL available data for final prediction - Uses full dataset
rf.fit(X, y)

pred_path = os.path.join(data_folder, pred_file)

if not os.path.exists(pred_path):
    raise FileNotFoundError('Prediction File Missing: {pred_file}')

pred_df = pd.read_csv(pred_path)

home_teams = pred_df['Home']
away_teams = pred_df['Away']

X_pred = pred_df[features]

if X_pred.isna().sum().sum() > 0:
    print('Warning: Missing Values')
    X_pred = X_pred.fillna(0)

y_pred_prob = rf.predict_proba(X_pred)
y_pred_class = rf.predict(X_pred)
p_home_win = y_pred_prob[:, 1]

# Structure of results dataframe for weekly predictions
results = pd.DataFrame({
    'Home_Team':home_teams,
    'Away_Team': away_teams,
    'Home_Prob': p_home_win,
    'Predicted_Winner': ['Home' if p == 1 else 'Away' for p in y_pred_class],
    'Home_Or_Away': ['Home' if p == 1 else 'Away' for p in y_pred_class]
})

results['Predicted_Winning_Team']= [
    row['Home_Team'] if row['Predicted_Winner'] == 'Home' else row['Away_Team']
    for _, row in results.iterrows()
]

print(f'\n+--------------------Week {current_week} Game Predictions:---------------------+')
#print('Week 3 Game Predictions:') <-- 24 chars
print(results[['Home_Team', 'Away_Team', 'Home_Prob', 'Predicted_Winning_Team', 'Home_Or_Away']].to_string(index=False, float_format='%.2f'))
print('+-----------------------------------------------------------------+')
#y_pred = rf.predict(X_test)

# Add weekly prediction results history output file
output_file = os.path.join(data_folder, f'Week{current_week}_Predictions_History.csv')
results.to_csv(output_file, index=False)
print(f"\n✓ Week {current_week} predictions saved to: {output_file}\n")


print("\nHome Win Rate:")
print((full_train_df['Outcome'] == 1).mean())

print("\nBalanced Home Win Rate:")
print((balanced_df['Outcome'] == 1).mean())

# Maybe add feature importance logic

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feat_imp.head(10))





# Week 3 Predictions: Went 11/16 (68.8%) - hell yah
# Week 4 Predictions: Went 10/15 (66.7%) - not bad (tie kinda screwed me)
# Week 5 Predictions: Went 6/14 (42.9%) - Not good. Weird week. Missed a lot of close games (1-4 pts each)