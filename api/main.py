from flask import Flask, redirect, request, url_for, render_template, jsonify
import pandas as pd
import predictor


app = Flask(__name__, 
            static_folder="static",
            template_folder="templates")


# At the top of your Flask app, load the player names once
def load_player_names(): 
    # Load your CSV (same one you used for ML)
    df = pd.read_csv("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv") 
    player_names = df['name'].unique().tolist()
    return sorted(player_names)  # Sort alphabetically

# Load once when app starts
ALL_PLAYERS = load_player_names()

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/api/data")
def get_data():
    return jsonify({"message": "API is working"})


@app.route('/search_players')
def search_players():
    query = request.args.get('q', '').lower()
    if len(query) < 2:
        return jsonify({'players': []})  # Use jsonify
    
    matching = [name for name in ALL_PLAYERS if query in name.lower()]
    return jsonify({'players': matching[:10]})  # Use jsonify



#----------------------------------------
pred_model, data = predictor.get_model_and_data()

@app.route('/api/predict', methods=['POST'])
def predict_player():
    # Handle both JSON and form data
    if request.is_json:
        # JSON request (from JavaScript fetch, curl, etc.)
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No JSON data provided'}), 400
            player_name = data.get('player_name')
        except Exception as e:
            return jsonify({'error': 'Invalid JSON format'}), 400
    else:
        # Form data (from HTML form submission)
        player_name = request.form.get('player_name')
    
    # Validate player name
    if not player_name:
        if request.is_json:
            return jsonify({'error': 'Player name required'}), 400
        else:
            return render_template("results.html", error="Player name required")
    
    # Strip whitespace and handle case sensitivity
    player_name = player_name.strip()
    
    # Check if player exists in your dataset
    matching_players = [name for name in ALL_PLAYERS if name.lower() == player_name.lower()]
    
    if not matching_players:
        # Try partial match to provide better error message
        partial_matches = [name for name in ALL_PLAYERS if player_name.lower() in name.lower()]
        
        if request.is_json:
            error_response = {
                'error': f'Player "{player_name}" not found in database',
                'suggestion': 'Use the search endpoint to find valid player names'
            }
            if partial_matches:
                error_response['similar_players'] = partial_matches[:5] # type: ignore
            return jsonify(error_response), 404
        else:
            return render_template("results.html", 
                                 error=f'Player "{player_name}" not found in database',
                                 similar_players=partial_matches[:5])
    
    # Use the exact player name from the dataset
    exact_player_name = matching_players[0]
    
    try:
        # Get model and data
        model, data = predictor.get_model_and_data()
        
        # Run prediction
        prediction = predictor.predict_single_player(exact_player_name, model, data)
        
        if request.is_json:
            return jsonify({
                'player_name': exact_player_name,
                'predicted_points': prediction,
                'success': True
            })
        else:
            return render_template("results.html",
                                 player_name=exact_player_name,
                                 predicted_points=prediction,
                                 success=True)
        
    except Exception as e:
        if request.is_json:
            return jsonify({'error': f'Prediction failed: {str(e)}', 'success': False}), 500
        else:
            return render_template("results.html", error=f'Prediction failed: {str(e)}')


# Commented out for Vercel to Run
# if __name__ == "__main__":
#     app.run(debug=True)
    

# @app.route("/login")
# def login():
#     return render_template()

# def user(usr):
#     return render_template()




# @app.route("/<name>")
# def user(name):
#     return f"Hello {name}"

# @app.route("/admin")
# def admin():
#     return redirect(url_for("home"))