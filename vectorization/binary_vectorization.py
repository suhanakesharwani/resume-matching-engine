import numpy as np

def binary_vectorize(jobs, vocabulary):
    """
    Convert job descriptions into binary vectors.

    Args:
        jobs (list): List of normalized job skills.
        vocabulary (list): Shared skill vocabulary.

    Returns:
        numpy array: Binary vectors for jobs.
    """

    # Create vocabulary index mapping
    vocab_index = {skill: i for i, skill in enumerate(vocabulary)}

    vectors = []

    # Create binary vector for each job
    for job in jobs:

        vec = np.zeros(len(vocabulary))

        for skill in job:
            if skill in vocab_index:
                vec[vocab_index[skill]] = 1

        vectors.append(vec)

    return np.array(vectors)