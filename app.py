import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# Load Preprocessed Data
@st.cache
def load_data():
    scaled_model_data = pd.read_csv('data/scaled_model_data.csv')
    original_data = pd.read_csv('data/spotifytoptracks.csv')
    return scaled_model_data, original_data

scaled_model_data, original_data = load_data()

# Ensure cosine and Euclidean similarity matrices are cached
@st.cache
def compute_similarity_matrices(data):
    # Compute cosine similarity
    cosine_similarity_matrix = cosine_similarity(data.values)
    cosine_sim_df = pd.DataFrame(
        cosine_similarity_matrix,
        index=original_data['Track Name'],
        columns=original_data['Track Name']
    )
    # Compute Euclidean similarity
    euclidean_similarity_matrix = -euclidean_distances(data.values)
    euclidean_sim_df = pd.DataFrame(
        euclidean_similarity_matrix,
        index=original_data['Track Name'],
        columns=original_data['Track Name']
    )
    return cosine_sim_df, euclidean_sim_df

cosine_sim_df, euclidean_sim_df = compute_similarity_matrices(scaled_model_data)

# Recommendation Logic
def combined_recommendations(song_name, data, cosine_df, euclidean_df, recommendation_type='By Similarity', top_n=10):
    if recommendation_type == 'By Similarity':
        if song_name not in cosine_df.index or song_name not in euclidean_df.index:
            return f"Song '{song_name}' not found in the dataset."

        # Combine recommendations from cosine and Euclidean similarity matrices
        cosine_recs = cosine_df[song_name].sort_values(ascending=False).iloc[1:top_n//2 + 1]
        euclidean_recs = euclidean_df[song_name].sort_values(ascending=False).iloc[1:top_n//2 + 1]

        combined = pd.concat([cosine_recs, euclidean_recs]).groupby(level=0).mean().sort_values(ascending=False)
        combined = combined.head(top_n)

        return combined.reset_index().rename(columns={'index': 'Track Name', 0: 'Similarity Score'})

    elif recommendation_type == 'By Artist':
        if song_name not in data['Track Name'].values:
            return f"Song '{song_name}' not found in the dataset."

        artist = data.loc[data['Track Name'] == song_name, 'Artist'].values[0]
        artist_songs = data[data['Artist'] == artist]

        recommendations = []
        for track in artist_songs['Track Name']:
            if track in cosine_df.index:
                similarity_score = cosine_df[song_name][track]
                recommendations.append((track, similarity_score))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(recommendations, columns=['Track Name', 'Similarity Score'])

    elif recommendation_type == 'By Genre':
        if song_name not in data['Track Name'].values:
            return f"Song '{song_name}' not found in the dataset."

        genre = data.loc[data['Track Name'] == song_name, 'Genre'].values[0]
        genre_songs = data[data['Genre'] == genre]

        recommendations = []
        for track in genre_songs['Track Name']:
            if track in cosine_df.index:
                similarity_score = cosine_df[song_name][track]
                recommendations.append((track, similarity_score))

        recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(recommendations, columns=['Track Name', 'Similarity Score'])

    else:
        return "Invalid recommendation type. Choose from 'By Similarity', 'By Artist', or 'By Genre'."

# Streamlit App UI
st.title("Music Recommendation System")

# Song Input
song_name = st.text_input("Enter a song name:")

# Recommendation Type Selection
recommendation_type = st.radio(
    "Select Recommendation Type:",
    ('By Similarity', 'By Artist', 'By Genre')
)

if song_name:
    st.write(f"Fetching {recommendation_type.lower()} recommendations for **{song_name}**...")
    recommendations = combined_recommendations(
        song_name=song_name,
        data=original_data,
        cosine_df=cosine_sim_df,
        euclidean_df=euclidean_sim_df,
        recommendation_type=recommendation_type,
        top_n=10
    )

    if isinstance(recommendations, str):  # Error message
        st.write(recommendations)
    else:
        st.write(f"### Recommendations ({recommendation_type})")
        st.dataframe(recommendations)
