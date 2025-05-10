# movie-recommendation-system
{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNhqRtmaP5oOSLhj4gs5WQC",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/ishwaryaav/movie-recommendation-system/blob/main/Untitled7.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Downgrade numpy and install surprise properly\n",
        "!pip install numpy==1.24.4\n",
        "!pip install scikit-surprise"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a4D8YkHivBiY",
        "outputId": "638745cd-f2a0-498f-c76f-690257c49714"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: numpy==1.24.4 in /usr/local/lib/python3.11/dist-packages (1.24.4)\n",
            "Requirement already satisfied: scikit-surprise in /usr/local/lib/python3.11/dist-packages (1.1.4)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.4.2)\n",
            "Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.24.4)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.15.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install scikit-surprise"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "mx37etgftw6x",
        "outputId": "aa9cfee6-9304-41f7-e042-b6780654da0b"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: scikit-surprise in /usr/local/lib/python3.11/dist-packages (1.1.4)\n",
            "Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.4.2)\n",
            "Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.24.4)\n",
            "Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.11/dist-packages (from scikit-surprise) (1.15.2)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Import libraries\n",
        "import pandas as pd\n",
        "from sklearn.metrics.pairwise import cosine_similarity\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "from surprise import Dataset, Reader, SVD\n",
        "from surprise.model_selection import train_test_split\n",
        "from surprise.accuracy import rmse\n",
        "\n",
        "# Load the CSV files\n",
        "movies = pd.read_csv('movies.csv')\n",
        "ratings = pd.read_csv('ratings.csv')\n",
        "\n",
        "# Merge ratings with movie titles\n",
        "data = pd.merge(ratings, movies, on='movieId')\n",
        "\n",
        "# ----------------------------\n",
        "# CONTENT-BASED FILTERING\n",
        "# ----------------------------\n",
        "# Preprocess genres\n",
        "movies['genres'] = movies['genres'].str.replace('|', ' ')\n",
        "vectorizer = CountVectorizer()\n",
        "genre_matrix = vectorizer.fit_transform(movies['genres'])\n",
        "\n",
        "# Compute similarity\n",
        "cos_sim = cosine_similarity(genre_matrix)\n",
        "\n",
        "# Function to get similar movies\n",
        "def get_similar_movies(movie_title, top_n=5):\n",
        "    index = movies[movies['title'] == movie_title].index[0]\n",
        "    similar_scores = list(enumerate(cos_sim[index]))\n",
        "    sorted_scores = sorted(similar_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]\n",
        "    similar_movies = [movies.iloc[i[0]]['title'] for i in sorted_scores]\n",
        "    return similar_movies\n",
        "\n",
        "print(\"🎬 Content-Based Recommendations for 'Toy Story (1995)':\")\n",
        "print(get_similar_movies('Toy Story (1995)'))\n",
        "\n",
        "# ----------------------------\n",
        "# COLLABORATIVE FILTERING\n",
        "# ----------------------------\n",
        "# Load data for surprise\n",
        "reader = Reader(rating_scale=(0.5, 5.0))\n",
        "data_surprise = Dataset.load_from_df(data[['userId', 'movieId', 'rating']], reader)\n",
        "\n",
        "trainset, testset = train_test_split(data_surprise, test_size=0.2, random_state=42)\n",
        "\n",
        "# Train model using SVD\n",
        "model = SVD()\n",
        "model.fit(trainset)\n",
        "\n",
        "# Predict and evaluate\n",
        "predictions = model.test(testset)\n",
        "print(\"\\n📊 Collaborative Filtering RMSE:\")\n",
        "rmse(predictions)\n",
        "\n",
        "# Recommend for user\n",
        "def recommend_movies_for_user(user_id, top_n=5):\n",
        "    movie_ids = data['movieId'].unique()\n",
        "    predictions = [model.predict(user_id, movie_id) for movie_id in movie_ids]\n",
        "    predictions.sort(key=lambda x: x.est, reverse=True)\n",
        "    top_movies = [movies[movies['movieId'] == pred.iid]['title'].values[0] for pred in predictions[:top_n]]\n",
        "    return top_movies\n",
        "\n",
        "print(f\"\\n🎯 Top 5 Movie Recommendations for User ID 1:\")\n",
        "print(recommend_movies_for_user(1))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "C5cknDaAubZe",
        "outputId": "347e6da6-f8e9-4ef4-9ace-b3be29b0d7ea"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "🎬 Content-Based Recommendations for 'Toy Story (1995)':\n",
            "['Antz (1998)', 'Toy Story 2 (1999)', 'Adventures of Rocky and Bullwinkle, The (2000)', \"Emperor's New Groove, The (2000)\", 'Monsters, Inc. (2001)']\n",
            "\n",
            "📊 Collaborative Filtering RMSE:\n",
            "RMSE: 0.8686\n",
            "\n",
            "🎯 Top 5 Movie Recommendations for User ID 1:\n",
            "['Goodfellas (1990)', 'Monty Python and the Holy Grail (1975)', 'Godfather, The (1972)', 'Grand Day Out with Wallace and Gromit, A (1989)', 'Godfather: Part II, The (1974)']\n"
          ]
        }
      ]
    }
  ]
}
