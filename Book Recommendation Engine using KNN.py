# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Load and clean the dataset
books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")

# Preprocess the data
user_counts = ratings['User-ID'].value_counts()
book_counts = ratings['ISBN'].value_counts()

# Filter users with at least 200 ratings and books with at least 100 ratings
filtered_users = user_counts[user_counts >= 200].index
filtered_books = book_counts[book_counts >= 100].index

filtered_ratings = ratings[(ratings('User-ID').isin(filtered_users))] & [ratings('ISBN').isin(filtered_books)]

# Pivot the ratings data into a user-book matrix
book_matrix = filtered_ratings.pivot(index='User-ID', columns='ISBN', values='Book-Rating')

# Fill missing values with 0
book_matrix = book_matrix.fillna(0)

# Build a K-Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute', n_jobs=-1)
knn.fit(book_matrix.values.T)

# Create a function to get book recommendations
def get_recommends(book_title):
    book_index = book_matrix.columns.get_loc(book_title)
    distances, indices = knn.kneighbors(book_matrix.values.T[book_index].reshape(1, -1), 6)

    recommended_books = [book_title]
    for i in range(1, 6):
        recommended_book = book_matrix.columns[indices[0][i]]
        distance = distances[0][i]
        recommended_books.append([recommended_book, distance])

    return recommended_books
result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(result)
