# Import Necessary Module
from sklearn.feature_extraction.text import TfidfVectorizer


# Function to Generate TF-IDF Vectors
def tfidf_vectorize(resumes):
    """
    Convert normalized resume skills into TF-IDF vectors.

    Args:
        resumes (list): List of normalized resume skill lists.

    Returns:
        tuple: TF-IDF matrix and fitted vectorizer.
    """

    # Convert skill lists into text documents
    docs = [" ".join(r) for r in resumes]

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform documents
    X = vectorizer.fit_transform(docs)

    # Return TF-IDF vectors and vectorizer
    return X, vectorizer