# PL Predicts

A web app that lets you search any English Premier League player and predict how many
fantasy points they would accumulate over a 5-game period.

**Tech used:** HTML, CSS, JavaScript, Python, scikit-learn, pandas, Flask

## Run Locally

### Prerequisites

- Python 3.11 or newer
- git
- An internet connection (the first prediction downloads the training dataset from GitHub)

### Setup

```bash
git clone <your-fork-url> PremPredictz
cd PremPredictz

# create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Run

```bash
python api/main.py
```

Then open http://127.0.0.1:5000 in your browser.

### Notes

- The home page loads instantly. The **first** request that needs the model
  (submitting a player, or `/search_players`) downloads `merged_gw.csv` and trains
  the Random Forest — expect a one-time pause of roughly 15-40 seconds. Subsequent
  predictions are fast; the model is cached in memory until the server restarts.
- `GET /health` returns `{"status": "ok"}` for a quick check that the server is up.
- Press `Ctrl+C` to stop the server.

## How It's Made

The frontend is plain HTML/CSS/JavaScript ([api/templates/](api/templates/),
[api/static/](api/static/)). The backend is a Flask app ([api/main.py](api/main.py))
that serves the pages and exposes a `/api/predict` endpoint. Predictions come from a
`RandomForestRegressor` ([api/predictor.py](api/predictor.py)) trained on **2023/24**
Premier League gameweek data (29,725 rows across 869 players, all 38 gameweeks).

## Lessons Learned

* **Random Forest Regression**: Used ensemble learning to predict a player's total
  fantasy points over their next 5 games.
* **Data Processing**: Cleaned and preprocessed raw FPL gameweek data, handling
  missing values and selecting relevant features from historical statistics.
* **Feature Engineering**: Created derived features such as 6-game rolling averages
  for points, goals, assists, minutes, and clean sheets to capture player form.
* **Full-Stack Integration**: Built an end-to-end pipeline connecting a Flask backend
  with a lightweight vanilla-JS frontend for on-demand player performance forecasting.

## Deploy (Vercel)

Deployment config lives in [vercel.json](vercel.json):

- `@vercel/python` builds [api/main.py](api/main.py) as a serverless function.
- All routes are rewritten to that function; Flask serves the pages and static assets.
- `includeFiles: "api/**"` ensures the templates and static files are bundled.

Push to a Git repo connected to Vercel, or run `vercel` from the CLI. `vercel dev`
runs the same setup locally.

**Caveat:** `pandas` + `scikit-learn` (plus its `scipy`/`numpy` dependencies) unzip
to close to Vercel's 250 MB serverless-function limit, so the build can fail on size.
If that happens, precompute the predictions offline (write them to a JSON/CSV the app
just looks up) and drop `scikit-learn` from the runtime dependencies.
