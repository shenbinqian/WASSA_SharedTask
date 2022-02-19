# -*- coding: UTF-8 -*-

from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import numpy as np


def embedding(texts, type='USE'):
    if type == 'USE':
        print('Embedding processing...')
        embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/4")
        embeddings = np.array(list(embed(texts).values())).squeeze()
        return embeddings
    if type == 'SBERT':
        print('Embedding processing...')
        model = SentenceTransformer('all-mpnet-base-v2')
        return model.encode(texts)
    else:
        print('Please check your embedding type!')
        print('It should be "USE" or "SBERT".')