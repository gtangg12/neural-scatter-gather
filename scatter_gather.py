import random
import numpy as np

from util import pca, cluster_complete


def dot_norm(a, b):
    """ Cosine similarity """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_nuance(embeddings):
    """ Sum of principal components is used because downstream pairwise
        comparison can be implemented as norm_dot of pairwise sums
    """
    principal_components = pca(embeddings, 16)
    return np.sum(principal_components)


class Query:
    def __init__(self, text):
        embeddings = embed_sentences(text)
        self.sentence_embeddings = embeddings
        self.centroid = np.sum(embeddings)
        self.nuance = compute_nuance(embeddings)

    def compare(cluster):
        """ Returns score in [-2, 2] indicating quality of match """
        return dot_norm(self.centroid, cluster.centroid) + \
            dot_norm(self.nuance, cluster.nuance)


class Cluster:
    def __init__(self, embeddings, document_index, centroid):
        self.embeddings = embeddings
        self.document_index = document_index
        self.centroid = centroid
        self.nuance = compute_nuance(centroid)


class Layer:
    def __init__(self, embeddings, document_index, gather_threshold=None):
        """
        Inputs:
            gather_threshold: num clusters to select in gather

        Variables:
            n, k: number of documents, clusters
        """
        self.embeddings = embeddings
        self.document_index = document_index
        self.n = embeddings.shape[0]
        self.k = sqrt(n)
        self.clusters = []
        if not gather_threshold:
            gather_threshold = sqrt(k)

    def scatter(self, clustering_algorithm):
        """
        """
        centroids, nearest_centroid = clustering_algorithm(embeddings, k)
        for i in range(self.k):
            cluster_index = np.where(nearest_centroid == i)
            self.clusters.append(Cluster(embeddings[cluster_index],
                                         document_index[cluster_index],
                                         centroids[i]))

    def gather(self, query):
        """
        """
        scores = np.zeros(k)
        for cluster in self.clusters:
            scores[i] = query.compare(cluster)

        top_clusters_index = \
            np.argpartition(a, -gather_threshold)[-gather_threshold:]

        embedddings, document_index = [], []
        for i in top_clusters_index:
            embedddings.append(self.clusters[i].embeddings)
            document_index.append(self.clusters[i].document_index)

        return np.concatenate(embedddings), np.concatenate(document_index)


def scatter_gather(documents, query, depth=3, topk=5):
    """
    Inputs:
        documents: DocumentList
        depth: number of layers to recursively create
        topk: number of relevant documents to return
    """
    embeddings = np.array([d['embedding'] for d in documents])
    document_index = np.arange(len(documents))

    init_layer = Layer(embeddings, document_index)
    init_layer.scatter(cluster_complete)

    cur_layer = init_layer
    for i in range(depth - 1):
        next_layer = Layer(*cur_layer.gather(query))
        next_layer.scatter(cluster_complete)
        cur_layer = next_layer

    document_index_to_sample = []
    for cluster in cur_layer.clusters:
        document_index_to_sample.extend(cluster.document_index)

    ret_document_index = random.sample(document_index_to_sample, 5)
    return [documents[i] for i in ret_document_index]


def scatter_gather_all(dataset, latent_dataset, k=5):
    """ Return top 5 nearest neighbors for each embedding query from database

        neighbors: N x k array of top k document indicies, where N is number of queries
    """
    neighbors = []
    for document in dataset:
        neighbors.append(
            scatter_gather(latent_dataset.documents, Query(document['text'])))
    return np.array(neighbors)
