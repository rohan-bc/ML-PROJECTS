Movie Recommendation System
This project implements a Movie Recommendation System using collaborative filtering with the Singular Value Decomposition (SVD) algorithm. It uses the surprise library for building the recommendation model and matplotlib for visualizing the distribution of ratings.

Features
Load and preprocess movie data
Build and train a recommendation model
Generate movie recommendations for users
Visualize the distribution of ratings.


Python Libraries: pandas, scikit-learn, surprise, matplotlib.

Installation
Clone the Repository
git clone https://github.com/rohan-bc/ML-PROJECTS.git.

Install Required Libraries.

Ensure you have pip installed, then run:
pip install pandas scikit-learn surprise matplotlib.

Download Datasets
Download the ratings.csv and movies.csv files from MovieLens and place them in the project directory.

Prepare Data : 
Ensure that ratings.csv and movies.csv are in the same directory as movie_recommender.py.

Run the Script 
Execute the script using:

python movier.py

View Results

The script will output the distribution of ratings as a histogram.
It will also print movie recommendations for a specific user (user ID 1 in this example). You can change the user_id variable in the script to test other users.

Code Explanation
Data Handling:
load_data(): Loads the ratings and movie data from CSV files.
plot_ratings_distribution(): Visualizes the distribution of ratings.

Model Building:
build_model(): Creates and trains an SVD model for collaborative filtering using the surprise library.

Recommendation Generation:
recommend_movies(): Generates and sorts movie recommendations based on predicted ratings.

Main Function:
The main() function orchestrates data loading, model training, and recommendation generation.
Contributing

Feel free to fork the repository and submit pull requests. For any issues or feature requests, open an issue on GitHub.
