# # Import Necessary Module
# from sklearn.metrics.pairwise import cosine_similarity


# # Function to Compare Resume and JD Vectors
# def compare(tfidf_vectors, binary_vectors):
#     """
#     Compute cosine similarity between resume TF-IDF vectors
#     and job description binary vectors.

#     Args:
#         tfidf_vectors (list or array): TF-IDF vectors of resumes.
#         binary_vectors (list or array): Binary vectors of job descriptions.

#     Returns:
#         array: Cosine similarity matrix.
#     """

#     # Calculate cosine similarity
#     sims = cosine_similarity(tfidf_vectors, binary_vectors)

#     # Return similarity scores
#     return sims

import numpy as np

def cosine(a, b):
    """
    Compute cosine similarity between two vectors.

    Args:
        a (array): Resume vector.
        b (array): Job vector.

    Returns:
        float: Cosine similarity score.
    """

    dot = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return dot / (norm_a * norm_b)


def compare(tfidf_vectors, binary_vectors):
    """
    Compare resume vectors with job vectors using cosine similarity.

    Args:
        tfidf_vectors (list): Resume TF-IDF vectors.
        binary_vectors (array): Job binary vectors.

    Returns:
        numpy array: Similarity matrix.
    """

    sims = []

    # Compute similarity for each resume-job pair
    for r in tfidf_vectors:

        row = []

        for j in binary_vectors:
            row.append(cosine(r, j))

        sims.append(row)

    return np.array(sims)