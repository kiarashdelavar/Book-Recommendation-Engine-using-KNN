import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('BX-Books.csv', sep=';', on_bad_lines='skip', encoding='latin-1')
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', on_bad_lines='skip', encoding='latin-1')

user_counts = ratings['User-ID'].value_counts()
book_counts = ratings['ISBN'].value_counts()

filtered_users = user_counts[user_counts >= 200].index
filtered_books = book_counts[book_counts >= 100].index

filtered_ratings = ratings[(ratings['User-ID'].isin(filtered_users)) & (ratings['ISBN'].isin(filtered_books))]

book_matrix = filtered_ratings.pivot_table(index='User-ID', columns='ISBN', values='Book-Rating').fillna(0)

knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute', n_jobs=-1)
knn.fit(book_matrix.values.T)

def get_recommends(book_title):
    if book_title not in books['Book-Title'].values:
        return f"Book titled '{book_title}' not found in the dataset."
    
    book_isbn = books.loc[books['Book-Title'] == book_title, 'ISBN'].values[0]
    
    if book_isbn not in book_matrix.columns:
        return f"Book titled '{book_title}' does not have sufficient ratings for recommendation."

    book_index = book_matrix.columns.get_loc(book_isbn)
    distances, indices = knn.kneighbors(book_matrix.values.T[book_index].reshape(1, -1), n_neighbors=6)

    recommended_books = [book_title]
    for i in range(1, 6):
        recommended_isbn = book_matrix.columns[indices[0][i]]
        recommended_book_title = books.loc[books['ISBN'] == recommended_isbn, 'Book-Title'].values[0]
        distance = distances[0][i]
        recommended_books.append([recommended_book_title, distance])

    return recommended_books

result = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(result)
