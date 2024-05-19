# Book Recommendation System Using K-Nearest Neighbors

This project implements a book recommendation system using the K-Nearest Neighbors (KNN) algorithm on the Book-Crossings dataset. The dataset comprises 1.1 million ratings (on a scale of 1-10) for 270,000 books by 90,000 users. The goal is to create a function that recommends books similar to a given book title.

Dataset Description

The Book-Crossings dataset contains:

    1- 1.1 million ratings
    2- 270,000 books
    3- 90,000 users

Methodology:
  
    1- Data Import and Cleaning: Load the dataset and remove users with fewer than 200 ratings and books with fewer than 100 ratings to ensure statistical significance.
    2- Model Development: Use the NearestNeighbors class from sklearn.neighbors to build a KNN model that determines the "closeness" of books based on user ratings.
    3- Recommendation Function: Implement a function get_recommends that takes a book title as input and returns a list of 5 similar books along with their distances from the input book.

Function: get_recommends

The get_recommends function takes a book title as an argument and returns recommendations in the following format:

    get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")

Output:

    [
        'The Queen of the Damned (Vampire Chronicles (Paperback))',
        [
            ['Catch 22', 0.793983519077301],
            ['The Witching Hour (Lives of the Mayfair Witches)', 0.7448656558990479],
            ['Interview with the Vampire', 0.7345068454742432],
            ['The Tale of the Body Thief (Vampire Chronicles (Paperback))', 0.5376338362693787],
            ['The Vampire Lestat (Vampire Chronicles, Book II)', 0.5178412199020386]
        ]
    ]

The output is a list where the first element is the input book title, and the second element is a list of five lists. Each sublist contains a recommended book and the distance from the input book.

Implementation:

    # Importing libraries
    import pandas as pd
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    # Load and clean data
    
    # Add your data loading and cleaning code here

    # Develop the model
    
    # Add your model training code here

    # Define the recommendation function
    
    def get_recommends(book_title):
    # Add your recommendation code here
    
    return

    # Test the function
    print(get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))"))

Place your data processing, model training, and recommendation logic between the provided cells to complete the implementation.

Instructions:

    1- Clone the repository.
    2- Load the Book-Crossings dataset.
    3- Clean the data by removing users with fewer than 200 ratings and books with fewer than 100 ratings.
    4- Train the KNN model using NearestNeighbors.
    5- Implement the get_recommends function.
    6- Test the function with various book titles to verify the recommendations.
