import random
import time
import numpy as np
import torch
from datasets import Dataset, load_dataset

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
    return np.sum(embed_sentences(document['text']))


def make_scatter_gather(document):
    return {
        'embedding': embed_document(document),
        'unbiased': document['domain'] in unbiased_news_sources,
    }


dataset = load_dataset('data/cc_news', split='train')

#dataset_processed = cc_news.map(make_scatter_gather, num_proc=256)
dataset_processed.save_to_disk("data/cc_news")
