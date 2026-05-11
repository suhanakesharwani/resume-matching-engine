import math
from collections import defaultdict

def tfidf_vectorize(resumes):
    """
    Convert resumes into TF-IDF vectors using manual implementation.

    Args:
        resumes (list): List of normalized resume skills.

    Returns:
        tuple: TF-IDF vectors and shared vocabulary.
    """

    # Remove duplicate skills from each resume
    resumes = [list(set(r)) for r in resumes]

    # Build vocabulary from all resumes
    vocabulary = set()

    for r in resumes:
        vocabulary.update(r)

    vocabulary = sorted(list(vocabulary))

    # Compute document frequency for each skill
    df = defaultdict(int)

    for skill in vocabulary:
        for r in resumes:
            if skill in r:
                df[skill] += 1

    N = len(resumes)

    vectors = []

    # Compute TF-IDF vector for each resume
    for r in resumes:

        vec = []

        for skill in vocabulary:

            tf = 1 if skill in r else 0
            idf = math.log(N / df[skill]) if df[skill] else 0

            vec.append(tf * idf)

        vectors.append(vec)

    return vectors, vocabulary