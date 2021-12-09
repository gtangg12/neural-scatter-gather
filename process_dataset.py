import os
import random
import time
import pickle
import glob

import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk

from util import embed_sentences


# Total of 'unbiased' 21553 articles
unbiased_news_sources = {
    'www.nytimes.com',
    'apnews.com',
    'www.npr.org',
    'www.bloomberg.com',
    'www.wsj.com',
    'www.cnn.com',
    'www.reuters.com',
    'www.latimes.com',
    'www.bbc.com',
}


def embed_document(document):
    """ Document embedding is sum of all sentence embeddings. """
    return np.sum(embed_sentences(document['text']), axis=0)


def make_scatter_gather(document):
    return {
        #'embedding': embed_document(document),
        'unbiased': document['domain'] in unbiased_news_sources,
    }

'''
dataset = load_dataset('cc_news', split='train')
print("Dataset Loaded!")
dataset_processed = dataset.map(make_scatter_gather, num_proc=1)
dataset_processed.save_to_disk("cc_news_unbiased")
'''

'''
dataset = load_from_disk('cc_news_unbiased')
print("Dataset Loaded!")
smaller_dataset = dataset.filter(lambda x: x['unbiased'] or random.random() < 0.075)
print(len(smaller_dataset))
dataset_processed = smaller_dataset.map(make_scatter_gather, num_proc=1)
dataset_processed.save_to_disk("cc_news_smaller_processed")
'''


def read_format(path):
    """ Read dataset format specified in save_format """
    batch = int(path.split('/')[-1])
    documents = pickle.load(open(path + '_doc.pl', 'rb'))
    embeddings = np.load(path + '_emb.npy')
    return batch, documents, embeddings


def save_format(dataset, path):
    """ Save dataset loaded as hugging face dataset into document pickle file
        and embedding np array file to interface with other platforms that
        cannot install datasets
    """
    documents, embeddings = [], []
    for document in dataset:
        embeddings.append(document['embedding'])
        del document['embedding']
        documents.append(document)

    pickle.dump(documents, open(path + '_doc.pl', 'wb'))
    # Numpy automatically adds extension
    np.save(path + '_emb', np.array(embeddings))


def check_dump(dataset, path, batch, batch_size=5000, last_batch=False):
    """ Dump dataset if accumulated more than batch_size elements to avoid
        memory slowdown
    """
    if not last_batch and len(dataset) < batch_size:
        return batch
    save_format(dataset, '{}/{:02d}'.format(path, batch))
    dataset.clear()
    return batch + 1


def save_train_latent_split(dataset, path, batch_size=5000):
    """ Split datset into train (half of unbiased) and latent database
        (about same number of unbiased as train and 5x more random)
    """
    os.makedirs(f'{path}/train', exist_ok=True)
    os.makedirs(f'{path}/latent', exist_ok=True)

    train_batches, latent_batches = 0, 0

    train, latent = [], []
    for i, document in enumerate(tqdm(dataset)):
        if document['unbiased'] and random.random() < 0.5:
            train.append(document)
        else:
            latent.append(document)

        train_batches = check_dump(
            train, f'{path}/train', train_batches)
        latent_batches = check_dump(
            latent, f'{path}/latent', latent_batches)

    if len(train):
        train_batches = check_dump(
            train, f'{path}/train', train_batches, last_batch=True)
    if len(latent):
        latent_batches = check_dump(
            latent, f'{path}/latent', latent_batches, last_batch=True)

    print(f'{train_batches} train batches written')
    print(f'{latent_batches} latent batches written')


class DocumentList():
    """ Store document in separate list so memory can be chuncked """
    def __init__(self, documents, batch_size=5000):
        self.documents = documents
        self.batch_size = batch_size

    def __len__(self):
        return (len(self.documents) - 1) * self.batch_size \
            + len(self.documents[-1])

    def __getitem__(self, idx):
        return documents[idx // self.batch_size][idx % self.batch_size]


def main():
    '''
    dataset = load_from_disk('data/cc_news_smaller_processed')
    print('Dataset Loaded!')
    save_train_latent_split(dataset, 'data/cc_news_smaller_processed_split')
    '''
    '''
    cnt = 0
    train = glob.glob('data/cc_news_smaller_processed_split/train/*')
    latent = glob.glob('data/cc_news_smaller_processed_split/latent/*')

    all_embeddings = []
    all_documents = []
    for t in latent:
        if t.endswith('.npy'):
            continue
        #print(t)
        name = t.rstrip('_doc.pl')
        #print(name)
        batch, documents, embeddings = read_format(name)
        #print(batch)
        #print(len(documents))
        #print(documents[0])
        #print(len(embeddings))
        #print(embeddings[0])
        all_embeddings.append(embeddings)
        all_documents.append(documents)

    all_embeddings = np.concatenate(all_embeddings)
    print(all_embeddings.shape)
    print(len(all_documents))
    document_db = DocumentList(all_documents)
    print(len(document_db))
    '''


if __name__ == '__main__':
    main()
