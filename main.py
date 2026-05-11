from preprocessing.data_preprocessing import preprocess
from vectorization.tfidf_vectorization import tfidf_vectorize
from vectorization.binary_vectorization import binary_vectorize
from similarity.comparison import compare
import numpy as np

def main():

    names,resumes,jd_names,jobs = preprocess()

    tfidf_vectors,vectorizer = tfidf_vectorize(resumes)

    binary_vectors = binary_vectorize(jobs,vectorizer)

    similarity_matrix = compare(tfidf_vectors,binary_vectors)

    for j, jd in enumerate(jd_names):

        scores = similarity_matrix[:,j]

        top3 = np.argsort(scores)[::-1][:3]

        print(f"\n{jd}")
        for i in top3:
            print(f"{names[i]} ({scores[i]:.3f})")

if __name__ == "__main__":
    main()