import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Load your preprocessed data
tv_data = pd.read_csv("top500_tv_shows.csv")

# TF-IDF encoding
vectorizer = TfidfVectorizer(token_pattern='[a-zA-Z]+')
genre_matrix = vectorizer.fit_transform(tv_data['genres'].fillna(''))
similarity_matrix = cosine_similarity(genre_matrix)

# Recommendation function
def recommend_show(title, n=10):
    if title not in tv_data['primaryTitle'].values:
        return pd.DataFrame(columns=['primaryTitle', 'genres', 'averageRating'])
    
    idx = tv_data[tv_data['primaryTitle'] == title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in scores[1:n+1]]
    return tv_data.iloc[top_indices][['primaryTitle', 'genres', 'averageRating']]

# Streamlit UI
st.title("🎬 TV Show Recommendation System")
user_input = st.text_input("Enter your favorite TV show:")

if user_input:
    st.subheader(f"Recommendations for '{user_input}'")
    recommendations = recommend_show(user_input, n=6)  # choose how many to show
    if recommendations.empty:
        st.write("Sorry, that show is not in the dataset.")
    else:
        # Create rows of 3 columns each
        for i in range(0, len(recommendations), 3):
            cols = st.columns(3)
            for col, (_, row) in zip(cols, recommendations.iloc[i:i+3].iterrows()):
                with col:
                    st.markdown(f"### {row['primaryTitle']}")
                    st.write(f"**Genres:** {row['genres']}")
                    st.write(f"⭐ {row['averageRating']}")
                