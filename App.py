from flask import Flask, request, jsonify
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Step 1: Load Preprocessed Data and Similarity Matrices
scaled_model_data = pd.read_csv('data/scaled_model_data.csv')
cosine_sim_df = pd.read_csv('data/cosine_similarity_matrix.csv', index_col=0)
euclidean_sim_df = pd.read_csv('data/euclidean_similarity_matrix.csv', index_col=0)

# Step 2: Recommendation Function
def combined_recommendations_no_duplicates(song_name, cosine_df, euclidean_df, top_n=10):
    """
    Combines recommendations from Cosine and Euclidean Similarity matrices, removing duplicates.

    Parameters:
    - song_name (str): Name of the song to base recommendations on.
    - cosine_df (DataFrame): Cosine similarity matrix.
    - euclidean_df (DataFrame): Euclidean similarity matrix.
    - top_n (int): Number of recommendations to return.

    Returns:
    - DataFrame: Combined recommendations with scores.
    """
    if song_name not in cosine_df.index or song_name not in euclidean_df.index:
        return f"Song '{song_name}' not found in the dataset."

    # Get recommendations from both similarity matrices
    cosine_recs = cosine_df[song_name].sort_values(ascending=False).iloc[1:6]
    euclidean_recs = euclidean_df[song_name].sort_values(ascending=False).iloc[1:6]

    # Combine and remove duplicates
    combined = pd.concat([cosine_recs, euclidean_recs]).groupby(level=0).mean().sort_values(ascending=False)
    combined = combined.head(top_n)

    return combined.reset_index().rename(columns={'index': 'Track Name', 0: 'Similarity Score'})

# Step 3: Define API Endpoint
@app.route('/recommend', methods=['GET'])
def recommend():
    """
    API Endpoint for song recommendations.

    Query Parameter:
    - song: Name of the input song.

    Returns:
    - JSON response with recommendations or error message.
    """
    song_name = request.args.get('song')  # Get the song name from query parameters

    if not song_name:
        return jsonify({"error": "Please provide a song name using the 'song' query parameter."}), 400

    try:
        recommendations = combined_recommendations_no_duplicates(song_name, cosine_sim_df, euclidean_sim_df, top_n=10)
        if isinstance(recommendations, str):  # Song not found
            return jsonify({"error": recommendations}), 404

        # Format recommendations as JSON
        recommendations_list = recommendations.to_dict(orient='records')
        return jsonify({"input_song": song_name, "recommendations": recommendations_list}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# Step 4: Run the Flask App (Uncomment this if running locally)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
