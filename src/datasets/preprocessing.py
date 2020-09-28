from sklearn.feature_extraction.text import TfidfTransformer

import torch
import numpy as np


def compute_tfidf_weights(data,vocab_size):
    """ Compute the Tf-idf weights (based on idf vector computed from dataset)."""

    transformer = TfidfTransformer()

    # fit idf vector on dataset
    counts = np.zeros((len(data), vocab_size), dtype=np.int64)
    for i, row in enumerate(data):
        counts_sample = torch.bincount(row['text'])
        counts[i, :len(counts_sample)] = counts_sample.cpu().data.numpy()
    tfidf = transformer.fit_transform(counts)

    for i, row in enumerate(data):
        row['weight'] = torch.tensor(tfidf[i, row['text']].toarray().astype(np.float32).flatten())
