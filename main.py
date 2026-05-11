from preprocessing.data_preprocessing import preprocess
from vectorization.tfidf_vectorization import tfidf_vectorize
from vectorization.binary_vectorization import binary_vectorize
from similarity.comparison import compare
import numpy as np

def main():
    """
    Execute the Resume–JD matching pipeline.
    """

    # Preprocess candidate and job data
    names, resumes, jd_names, jobs = preprocess()

    # Create TF-IDF vectors for resumes
    tfidf_vectors, vocabulary = tfidf_vectorize(resumes)

    # Create binary vectors for job descriptions
    binary_vectors = binary_vectorize(jobs, vocabulary)

    # Compute similarity matrix
    similarity_matrix = compare(tfidf_vectors, binary_vectors)

    # Display Top 3 candidates for each job
    for j, jd in enumerate(jd_names):

        scores = similarity_matrix[:, j]

        top3 = np.argsort(scores)[::-1][:3]

        print(f"\n{jd}")

        for i in top3:
            print(f"{names[i]} ({scores[i]:.3f})")


if __name__ == "__main__":
    main()