from flask import Flask, request, render_template, jsonify

import predictor

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)


def get_all_players():
    """Player list, resolved lazily so a cold start doesn't do heavy work at import."""
    return predictor.get_all_player_names()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/api/data")
def get_data():
    return jsonify({"message": "API is working"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route('/search_players')
def search_players():
    query = request.args.get('q', '').lower()
    if len(query) < 2:
        return jsonify({'players': []})

    matching = [name for name in get_all_players() if query in name.lower()]
    return jsonify({'players': matching[:10]})


@app.route('/api/predict', methods=['POST'])
def predict_player():
    if request.is_json:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({'error': 'No JSON data provided'}), 400
        player_name = payload.get('player_name')
    else:
        player_name = request.form.get('player_name')

    if not player_name:
        if request.is_json:
            return jsonify({'error': 'Player name required'}), 400
        return render_template("results.html", error="Player name required")

    player_name = player_name.strip()

    try:
        all_players = get_all_players()
    except Exception as e:
        message = f'Model/data unavailable: {e}'
        if request.is_json:
            return jsonify({'error': message, 'success': False}), 503
        return render_template("results.html", error=message)

    matching_players = [name for name in all_players if name.lower() == player_name.lower()]

    if not matching_players:
        partial_matches = [name for name in all_players if player_name.lower() in name.lower()]

        if request.is_json:
            error_response = {
                'error': f'Player "{player_name}" not found in database',
                'suggestion': 'Use the search endpoint to find valid player names',
            }
            if partial_matches:
                error_response['similar_players'] = partial_matches[:5]  # type: ignore
            return jsonify(error_response), 404

        return render_template(
            "results.html",
            error=f'Player "{player_name}" not found in database',
            similar_players=partial_matches[:5],
        )

    exact_player_name = matching_players[0]

    try:
        model, data = predictor.get_model_and_data()
        prediction = predictor.predict_single_player(exact_player_name, model, data)

        if request.is_json:
            return jsonify({
                'player_name': exact_player_name,
                'predicted_points': prediction,
                'success': True,
            })
        return render_template(
            "results.html",
            player_name=exact_player_name,
            predicted_points=prediction,
            success=True,
        )
    except Exception as e:
        if request.is_json:
            return jsonify({'error': f'Prediction failed: {e}', 'success': False}), 500
        return render_template("results.html", error=f'Prediction failed: {e}')


# For local development only. Vercel imports `app` directly and ignores this.
if __name__ == "__main__":
    app.run(debug=True)
