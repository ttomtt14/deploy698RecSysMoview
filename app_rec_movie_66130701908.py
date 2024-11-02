
import streamlit as st
import pickle
from surprise import SVD

# Load data from the file
with open('66130701908recommendation_movie_svd.pkl', 'rb') as file:
    svd_model, movie_ratings, movies = pickle.load(file)

# Streamlit app
st.title("Movie Recommendation System")

# User input for selecting User ID
user_id = st.number_input("Enter User ID:", min_value=1, value=1, step=1)

# Filter out rated movies for the selected user
rated_user_movies = movie_ratings[movie_ratings['userId'] == user_id]['movieId'].values
unrated_movies = movies[~movies['movieId'].isin(rated_user_movies)]['movieId']

# Predict ratings for unrated movies
pred_rating = [svd_model.predict(user_id, movie_id) for movie_id in unrated_movies]
sorted_predictions = sorted(pred_rating, key=lambda x: x.est, reverse=True)

# Get top 10 movie recommendations
top_recommendations = sorted_predictions[:10]

# Display top recommendations
st.write(f"Top 10 movie recommendations for User {user_id}:")

for recommendation in top_recommendations:
    movie_title = movies[movies['movieId'] == recommendation.iid]['title'].values[0]
    st.write(f"{movie_title} (Estimated Rating: {recommendation.est})")

