import numpy as np
import h5py
import os
import requests
import tempfile
import time


# scann requires tensorflow
'''
import scann
with tempfile.TemporaryDirectory() as tmp:
    response = requests.get("http://ann-benchmarks.com/glove-100-angular.hdf5")
    loc = os.path.join(tmp, "glove.hdf5")
    with open(loc, 'wb') as f:
        f.write(response.content)

    glove_h5py = h5py.File(loc, "r")

print(list(glove_h5py.keys()))

num_test = 10
dataset = glove_h5py['train']
queries = glove_h5py['test'][:num_test]
print(dataset.shape)
print(queries.shape)


normalized_dataset = dataset / np.linalg.norm(dataset, axis=1)[:, np.newaxis]
# configure ScaNN as a tree - asymmetric hash hybrid with reordering
# anisotropic quantization as described in the paper; see README

start = time.time()
# use scann.scann_ops.build() to instead create a TensorFlow-compatible searcher
searcher = scann.scann_ops_pybind.builder(normalized_dataset, 10, "dot_product").tree(
    num_leaves=2000, num_leaves_to_search=100, training_sample_size=250000).score_ah(
    2, anisotropic_quantization_threshold=0.2).reorder(100).build()
end = time.time()
print("Time:", end - start)

start = time.time()
neighbors, distances = searcher.search_batched(queries)
end = time.time()

def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total / true_neighbors.size

# we are given top 100 neighbors in the ground truth, so select top 10
print("Recall:", compute_recall(neighbors, glove_h5py['neighbors'][:num_test, :10]))
print("Time:", end - start)
'''


# Embeddings BERT doesn't have equal contribution per vector entry
# Use embeddings provided by realm

import torch
import torch.nn
from sentence_transformers import SentenceTransformer

'''
tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = BertModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
'''
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
sentences = [
    "Three years later, the coffin was still full of Jello.",
    "The fish dreamed of escaping the fishbowl and into the toilet where he saw his friend go.",
    "the coffin was still full of Jello Three years later, ",
    "He found a leprechaun in his walnut shell."
]

'''
encoded_input = tokenizer(sentences, return_tensors='pt', padding=True)

output = model(**encoded_input)
print(output) # CLS last hidden state, (1, 768)
sentence_embeddings = output.pooler_output
'''
sentence_embeddings = model.encode(sentences)

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
for i in range(len(sentences)):
    print(i)
    for j in range(len(sentences)):
        print(cos(sentence_embeddings[i], sentence_embeddings[j]))
        #print(np.dot(sentence_embeddings[i], sentence_embeddings[j]))
    print()

# Realm embeddings; cant load will crash docker
'''
import tensorflow as tf

loaded = tf.saved_model.load('realm_pretrained/embedder', tags=[])
print(list(loaded.signatures.keys()))

infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)
'''
