import numpy as np
from collections import Counter

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    token_count = Counter(tokens)

    return np.array([token_count[word] for word in vocab], dtype=int)