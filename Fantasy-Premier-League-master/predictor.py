import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import TimeSeriesSplit

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"
data = pd.read_csv(url) #24/25 szn data

def load_data(df):
    
    # Create mapping dictionary
    position_mapping = {
        "GK": 1,
        "DEF": 2, 
        "MID": 3,
        "FWD": 4
    }

    # Apply mapping
    df["player_pos"] = df["position"].map(position_mapping).fillna(0)
    
    df.drop(columns=["transfers_in", "transfers_out","kickoff_time", "round", "starts", "fixture", "team_a_score", "team_h_score"], inplace=True)
    sorted_df = df.sort_values(by=['name','GW']) #Sorts data by player name(s)

    #Lists out the columns of the dataset
    return sorted_df

data = load_data(data) 



#Creating Featurezzz

feature_rows = []

# Group by player name instead of element
for player_name in data['name'].unique():
    player_data = data[data['name'] == player_name].copy()
    
    # We need at least 6 games for rolling averages + 5 future games for target
    if len(player_data) < 11:
        continue
    
    # For each possible prediction point
    for i in range(6, len(player_data) - 4):  # Start at game 6, end 5 games before last
        current_row = {}
        
        # Get the 6 previous games for features
        prev_6_games = player_data.iloc[i-6:i]
        
        # Get the next 5 games for target
        next_5_games = player_data.iloc[i+1:i+6]
        
        # Calculate rolling averages (features)
        current_row['avg_points_last_6'] = prev_6_games['total_points'].mean()
        current_row['avg_goals_last_6'] = prev_6_games['goals_scored'].mean()
        current_row['avg_assists_last_6'] = prev_6_games['assists'].mean()
        current_row['avg_minutes_last_6'] = prev_6_games['minutes'].mean()
        current_row['avg_clean_sheets_last_6'] = prev_6_games['clean_sheets'].mean()
        
        # Add other useful features
        current_row['player_pos'] = player_data.iloc[i]['player_pos']
        #current_row['team'] = player_data.iloc[i]['team']
        current_row['value'] = player_data.iloc[i]['value']
        #current_row['player_name'] = player_name
        current_row['current_gw'] = player_data.iloc[i]['GW']
        
        # Calculate target (next 5 games total points)
        current_row['target_points_next_5'] = next_5_games['total_points'].sum()
        
        feature_rows.append(current_row)

# Convert to DataFrame
features_df = pd.DataFrame(feature_rows)

# Separate X and y
feature_columns = ['avg_points_last_6', 'avg_goals_last_6', 'avg_assists_last_6', 
                  'avg_minutes_last_6', 'avg_clean_sheets_last_6', 'player_pos', 'value']

X = features_df[feature_columns]
y = features_df['target_points_next_5']


# print(f"Created {len(X)} training examples")
# print(f"Features: {feature_columns}")
# print(f"Target range: {y.min()} to {y.max()} points")


# print(f"Original X shape: {X.shape}")
# print(f"Original y shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#SCALING - Will uncomment later if needed
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#Debug Stuff
# print(f"X_train shape: {X_train.shape}")
# print(f"y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}")  
# print(f"y_test shape: {y_test.shape}")


#Hyper-parameter tuning
def tune_model(X_train, y_train):
    model = RandomForestRegressor(random_state=13)
    model.fit(X_train, y_train)
    return model

best_model = tune_model(X_train, y_train)

# Predictions and evaulate
#MAE should be as close to 0 as possible, and sqr error should be as close to 1 as possible
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    mean_abs_error = mean_absolute_error(y_test, prediction)
    mean_sqr_error = mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    return mean_abs_error, mean_sqr_error, r2


# Evaluate the model
# mae, mse, r2 = evaluate_model(best_model, X_test, y_test)
# print(f"Mean Absolute Error: {mae:.2f} points")
# print(f"Mean Squared Error: {mse:.2f}")
# print(f"R² Score: {r2:.3f}")


def predict_single_player(player_name, model, existing_data):
    # Use the same data we already loaded and processed
    player_data = existing_data[existing_data['name'] == player_name].copy()
    
    if len(player_data) < 6:
        return f"Not enough data for {player_name}"
    
    # Get their last 6 games from the existing data
    last_6_games = player_data.tail(6)
    
    # Rest of the function stays the same...
    features = {
        'avg_points_last_6': last_6_games['total_points'].mean(),
        'avg_goals_last_6': last_6_games['goals_scored'].mean(),
        'avg_assists_last_6': last_6_games['assists'].mean(),
        'avg_minutes_last_6': last_6_games['minutes'].mean(),
        'avg_clean_sheets_last_6': last_6_games['clean_sheets'].mean(),
        'player_pos': last_6_games['player_pos'].iloc[-1],
        'value': last_6_games['value'].iloc[-1]
    }
    
    feature_array = [[
        features['avg_points_last_6'], features['avg_goals_last_6'], 
        features['avg_assists_last_6'], features['avg_minutes_last_6'],
        features['avg_clean_sheets_last_6'], features['player_pos'], features['value']
    ]]
    
    prediction = model.predict(feature_array)[0]
    return round(prediction, 2)

# Test with existing data
# test_prediction = predict_single_player("Virgil van Dijk", best_model, data)



#Returns model and data to be used in the FLASK app
def get_model_and_data():
    return best_model, data

# print(f"Prediction: {test_prediction} points")






