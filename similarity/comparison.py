# Import Necessary Module
from sklearn.metrics.pairwise import cosine_similarity


# Function to Compare Resume and JD Vectors
def compare(tfidf_vectors, binary_vectors):
    """
    Compute cosine similarity between resume TF-IDF vectors
    and job description binary vectors.

    Args:
        tfidf_vectors (list or array): TF-IDF vectors of resumes.
        binary_vectors (list or array): Binary vectors of job descriptions.

    Returns:
        array: Cosine similarity matrix.
    """

    # Calculate cosine similarity
    sims = cosine_similarity(tfidf_vectors, binary_vectors)

    # Return similarity scores
    return sims