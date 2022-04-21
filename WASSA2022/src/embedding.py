# -*- coding: UTF-8 -*-

from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import numpy as np
from transformers import RobertaModel
from utils import tokenization
import torch


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


def extract_BERT_embeddings(train_texts, MAX_LENGTH=200, model_name='roberta-base', layer_num=12):
    train_ids, train_masks = tokenization(train_texts, MAX_LENGTH=MAX_LENGTH, model_name=model_name)
    # Load pre-trained model (weights)
    model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        outputs = model(train_ids, train_masks)
        hidden_states = outputs[2]

    # Concatenate the tensors for all layers. We use `stack` here to
    # create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)

    # return the chosen layer hidden states
    return token_embeddings[layer_num]
