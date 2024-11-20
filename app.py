import streamlit as st
import pandas as pd

# Load Preprocessed Data
@st.cache
def load_data():
    scaled_model_data = pd.read_csv('data/scaled_model_data.csv')
    cosine_sim_df = pd.read_csv('data/cosine_similarity_matrix.csv', index_col=0)
    euclidean_sim_df = pd.read_csv('data/euclidean_similarity_matrix.csv', index_col=0)
    return scaled_model_data, cosine_sim_df, euclidean_sim_df

scaled_model_data, cosine_sim_df, euclidean_sim_df = load_data()

# Recommendation Function
def combined_recommendations_no_duplicates(song_name, cosine_df, euclidean_df, top_n=10):
    if song_name not in cosine_df.index or song_name not in euclidean_df.index:
        return None
    cosine_recs = cosine_df[song_name].sort_values(ascending=False).iloc[1:6]
    euclidean_recs = euclidean_df[song_name].sort_values(ascending=False).iloc[1:6]
    combined = pd.concat([cosine_recs, euclidean_recs]).groupby(level=0).mean().sort_values(ascending=False)
    combined = combined.head(top_n)
    return combined.reset_index().rename(columns={'index': 'Track Name', 0: 'Similarity Score'})

# Streamlit App UI
st.title("Music Recommendation System")

# Song Input
song_name = st.text_input("Enter a song name:")

if song_name:
    st.write(f"Fetching recommendations for **{song_name}**...")
    recommendations = combined_recommendations_no_duplicates(song_name, cosine_sim_df, euclidean_sim_df, top_n=10)
    if recommendations is not None:
        st.write("### Recommended Songs")
        st.dataframe(recommendations)
    else:
        st.write(f"**{song_name}** not found in the dataset.")
