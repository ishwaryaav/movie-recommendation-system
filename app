# app.py

import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import streamlit as st

# Load data
@st.cache_data
def load_data():
    ratings_url = "https://raw.githubusercontent.com/sayakpaul/Recommendation-System-Tutorials/main/data/ratings.csv"
    movies_url = "https://raw.githubusercontent.com/sayakpaul/Recommendation-System-Tutorials/main/data/movies.csv"
    ratings = pd.read_csv(ratings_url)
    movies = pd.read_csv(movies_url)
    data = pd.merge(ratings, movies, on="movieId")
    return data, movies

# Train model
@st.cache_resource
def train_model(data):
    reader = Reader(rating_scale=(0.5, 5.0))
    dataset = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)
    trainset = dataset.build_full_trainset()
    model = SVD()
    model.fit(trainset)
    return model

# Get recommendations
def get_recommendations(user_id, data, movies, model, num=5):
    movie_ids = data['movieId'].unique()
    watched = data[data['userId'] == user_id]['movieId']
    unseen = [mid for mid in movie_ids if mid not in watched.values]
    
    predictions = [model.predict(user_id, mid) for mid in unseen]
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    top_movies = [pred.iid for pred in predictions[:num]]
    return movies[movies['movieId'].isin(top_movies)][['title', 'genres']]

# Streamlit UI
st.title("🎬 AI-Powered Movie Recommendation System")

data, movies = load_data()
model = train_model(data)

user_id = st.number_input("Enter User ID (between 1 and 600):", min_value=1, max_value=600, value=10)
if st.button("Get Recommendations"):
    try:
        recommendations = get_recommendations(user_id, data, movies, model)
        st.subheader("📽️ Recommended Movies:")
        st.table(recommendations)
    except:
        st.warning("User ID not found. Please enter a valid one.")
