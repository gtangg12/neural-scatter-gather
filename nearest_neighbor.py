import numpy as np
import scann

def knn(queries, dataset, k=5):
    """ Return top 5 nearest neighbors for each embedding query from database """
    normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]

    searcher = scann.scann_ops_pybind.builder(
            normalized_dataset, k, "dot_product"
        ).tree(
            num_leaves=2000, num_leaves_to_search=100, training_sample_size=70000
        ).score_ah(
            2, anisotropic_quantization_threshold=0.2
        ).reorder(100).build()

    neighbors, _ = searcher.search_batched(queries)
    return np.array(neighbors)
