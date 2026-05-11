# Import Necessary Module
import numpy as np


# Function to Create Binary Vectors for Job Descriptions
def binary_vectorize(jobs, vectorizer):
    """
    Convert normalized job skills into binary vectors
    using the shared vocabulary.

    Args:
        jobs (list): List of normalized job skill lists.
        vectorizer: Vectorizer containing vocabulary mapping.

    Returns:
        array: Binary vectors representing job descriptions.
    """

    # Get vocabulary mapping from vectorizer
    vocab = vectorizer.vocabulary_

    # Initialize list to store vectors
    vectors = []

    # Iterate over each job
    for job in jobs:

        # Create zero vector of vocabulary size
        vec = np.zeros(len(vocab))

        # Mark skill presence as 1
        for skill in job:
            if skill in vocab:
                vec[vocab[skill]] = 1

        # Append binary vector
        vectors.append(vec)

    # Return all job vectors as numpy array
    return np.array(vectors)