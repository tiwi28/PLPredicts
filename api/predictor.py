import re

import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 2023/24 Premier League season. Confirmed from the data itself, not just the URL:
# kickoffs run 2023-08-11 to 2024-05-19, and the 20 teams include Burnley, Luton
# and Sheffield Utd - the three sides promoted for that season.
DATA_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv"


def _season_from_url(url):
    """Derive a display season ("2023/24") from the dataset URL."""
    match = re.search(r"/(\d{4})-(\d{2})/", url)
    return f"{match.group(1)}/{match.group(2)}" if match else "Unknown season"


SEASON = _season_from_url(DATA_URL)

FEATURE_COLUMNS = [
    'avg_points_last_6', 'avg_goals_last_6', 'avg_assists_last_6',
    'avg_minutes_last_6', 'avg_clean_sheets_last_6', 'player_pos', 'value',
]

# Serverless-friendly cache. Nothing heavy runs at import time; the CSV download
# and model training happen once per warm instance, on the first request that
# needs them.
_CACHE = {}


def load_data(df):
    position_mapping = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    df["player_pos"] = df["position"].map(position_mapping).fillna(0)

    df.drop(
        columns=[
            "transfers_in", "transfers_out", "kickoff_time", "round",
            "starts", "fixture", "team_a_score", "team_h_score",
        ],
        inplace=True,
        errors="ignore",
    )
    return df.sort_values(by=['name', 'GW'])


def _build_features(data):
    feature_rows = []
    for player_name in data['name'].unique():
        player_data = data[data['name'] == player_name].copy()
        if len(player_data) < 11:
            continue

        for i in range(6, len(player_data) - 4):
            prev_6_games = player_data.iloc[i - 6:i]
            next_5_games = player_data.iloc[i + 1:i + 6]

            feature_rows.append({
                'avg_points_last_6': prev_6_games['total_points'].mean(),
                'avg_goals_last_6': prev_6_games['goals_scored'].mean(),
                'avg_assists_last_6': prev_6_games['assists'].mean(),
                'avg_minutes_last_6': prev_6_games['minutes'].mean(),
                'avg_clean_sheets_last_6': prev_6_games['clean_sheets'].mean(),
                'player_pos': player_data.iloc[i]['player_pos'],
                'value': player_data.iloc[i]['value'],
                'current_gw': player_data.iloc[i]['GW'],
                'target_points_next_5': next_5_games['total_points'].sum(),
            })

    return pd.DataFrame(feature_rows)


def _train():
    data = load_data(pd.read_csv(DATA_URL))
    features_df = _build_features(data)

    X = features_df[FEATURE_COLUMNS]
    y = features_df['target_points_next_5']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=13, n_jobs=-1)
    model.fit(X_train, y_train)

    return model, data


def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    return (
        mean_absolute_error(y_test, prediction),
        mean_squared_error(y_test, prediction),
        r2_score(y_test, prediction),
    )


def predict_single_player(player_name, model, existing_data):
    player_data = existing_data[existing_data['name'] == player_name].copy()

    if len(player_data) < 6:
        return f"Not enough data for {player_name}"

    last_6_games = player_data.tail(6)

    feature_array = pd.DataFrame([{
        'avg_points_last_6': last_6_games['total_points'].mean(),
        'avg_goals_last_6': last_6_games['goals_scored'].mean(),
        'avg_assists_last_6': last_6_games['assists'].mean(),
        'avg_minutes_last_6': last_6_games['minutes'].mean(),
        'avg_clean_sheets_last_6': last_6_games['clean_sheets'].mean(),
        'player_pos': last_6_games['player_pos'].iloc[-1],
        'value': last_6_games['value'].iloc[-1],
    }])[FEATURE_COLUMNS]

    return round(float(model.predict(feature_array)[0]), 2)


def get_model_and_data():
    """Return (model, processed_data), training once per instance."""
    if 'model' not in _CACHE:
        model, data = _train()
        _CACHE['model'] = model
        _CACHE['data'] = data
    return _CACHE['model'], _CACHE['data']


def get_all_player_names():
    _, data = get_model_and_data()
    return sorted(data['name'].unique().tolist())
