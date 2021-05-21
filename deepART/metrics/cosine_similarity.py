import numpy as np

def cos_sim(vecs_1, vecs_2):
    """
    vecs_1: numpy array or list
        A numpy matrix where each row represents a word vector, these 
        word vectors are those which you would want to obtain scores for.
        If vecs_1 is a list, each element would be a word vector.
    vecs_2: numpy array or list
        A numpy matrix where each row represents a word vector, these
        word vectors are those which you would want to score against.
        If vecs_2 is a list, each element would be a word vector.

    Scores each word vector in matrix (vecs_1) against each word vector in matrix (vecs_2), i.e.
    obtains a cosine similarity score for each word vector in matrix (vecs_1)

    The scores for each vector in matrix (vecs_1) are located in each row of the result of this function
    """

    if isinstance(vecs_1, list):
        vecs_1 = np.array(vecs_1)
    if isinstance(vecs_2, list):
        vecs_2 = np.array(vecs_2)
    if not isinstance(vecs_1, np.ndarray):
        raise ValueError("vecs_1 must be a numpy array or list")
    if not isinstance(vecs_2, np.ndarray):
        raise ValueError("vecs_1 must be a numpy array or list")

    numerator = np.dot(vecs_1, vecs_2.T)
    vecs_2_norm = np.linalg.norm(vecs_2, axis=1, ord=2)
    vecs_1_norm = np.linalg.norm(vecs_1, axis=1, ord=2)
    denominator = np.dot(vecs_1_norm.reshape((-1, 1)), vecs_2_norm.reshape((1,-1)))
    return numerator / denominator