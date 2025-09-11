from typing import List
import numpy as np


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    # Calculate cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate Euclidean distance between two embeddings"""
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    return np.linalg.norm(vec1 - vec2)
