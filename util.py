import numpy as np
import torch
import faiss
import spacy
from sentence_transformers import SentenceTransformer

spacy_processor = spacy.blank('en')
spacy_processor.add_pipe('sentencizer')

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


def embed_sentences(text):
    """ Given string text, output array of embeddings of text's sentences """
    sentences = text.split('\n')
    return model.encode(sentences)


def cluster_complete(embeddings, num_clusters):
    """
    Inputs:
        embeddings: N x d array
    Returns:
        centroids:
        nearest_centroid: N dim array of each embedding's assigned centroid
    """
    kmeans = faiss.Kmeans(d=embeddings.shape[1],
                          k=num_clusters,
                          niter=64,
                          gpu=torch.cuda.is_available())
    kmeans.train(embeddings)
    _, nearest_centroid = kmeans.index.search(x, 1)
    return kmeans.centroids, nearest_centroid


def pca(embeddings, rdim):
    """
    Inputs:
        embeddings: N x d array
        rdim: dimension to reduce to
    Returns:
        principal_components: rd x d torch tensor
    """
    d = embeddings.shape[1]
    pca = faiss.PCAMatrix(d, rdim)
    pca.train(embeddings)
    principal_components = faiss.vector_to_array(pca.A).reshape(rdim, d)
    return principal_components


def estimate_cluster_centers(embeddings):
    """ Esimate cluster centers using fast buckshot algorithm

    """
    pass


def assign_to_centroids():
    pass
