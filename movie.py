import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import matplotlib.pyplot as plt

# Load and preprocess data
def load_data():
    ratings = pd.read_csv('ratings.csv')  # userId, movieId, rating, timestamp
    movies = pd.read_csv('movies.csv')  # movieId, Title, Genres
    return ratings, movies

def plot_ratings_distribution(ratings):
    plt.figure(figsize=(10, 6))
    ratings['rating'].hist(bins=10, alpha=0.7)  # Update to match CSV column name
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    plt.show()

# Build recommendation model
def build_model(ratings):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)  # Update to match CSV column names
    trainset, testset = train_test_split(data, test_size=0.25)
    
    model = SVD()
    model.fit(trainset)
    
    predictions = model.test(testset)
    accuracy.rmse(predictions)
    
    return model, trainset

# Generate recommendations
def get_top_n(predictions, n=10):
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

def recommend_movies(user_id, model, trainset, n=10):
    predictions = []
    movie_ids = trainset.all_items()
    for movie_id in movie_ids:
        movie_id = trainset.to_raw_iid(movie_id)
        predictions.append((movie_id, model.predict(user_id, movie_id).est))
    
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

# Main function
def main():
    ratings, movies = load_data()
    plot_ratings_distribution(ratings)
    
    model, trainset = build_model(ratings)
    
    # Generate and print recommendations for a specific user
    user_id = str(1)  # Change this to a valid user ID from your dataset
    recommendations = recommend_movies(user_id=user_id, model=model, trainset=trainset, n=10)
    
    print(f"Recommendations for user {user_id}:")
    for movie_id, rating in recommendations:
        movie_title = movies[movies['movieId'] == int(movie_id)]['Title'].values
        print(f"{movie_title[0] if len(movie_title) > 0 else 'Unknown Movie'} - Estimated Rating: {rating:.2f}")

if __name__ == "__main__":
    main()
