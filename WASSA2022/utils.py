# -*- coding: UTF-8 -*-

import torch
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def get_dataframe(path="messages_train_ready_for_WS.tsv"):
    print('Start reading raw data...')
    raw = pd.read_table(path, delimiter='\t')
    df = raw[["empathy", "distress", "essay", "emotion"]]
    df['emotion'] = pd.Categorical(df.emotion)
    return df


def get_train_dev(X, train_size=0.9, test_size=0.1, random_state=0):
    X_train, X_test = train_test_split(X, train_size=train_size, test_size=test_size, random_state=random_state)
    return X_train, X_test


def oversample(X_train, y_train, random_state=27):
    sm = SMOTE(random_state=random_state)
    OS_Xtrain, OS_ytrain = sm.fit_resample(X_train, y_train)
    return OS_Xtrain, OS_ytrain


def select_feature(X, y, n_feature=6):
    # select most relative n features from X compared with y via chi squared test
    X_new = SelectKBest(chi2, k=n_feature).fit_transform(X, y)
    return X_new


def essay_article_similarity(embedding, essay_file='messages_train_ready_for_WS.tsv', article_file='articles_adobe_AMT.csv'):
    essays = pd.read_table(essay_file, delimiter='\t')
    articles = pd.read_csv(article_file)
    essay_embeddings = embedding(essays['essay'])
    article_embeddings = embedding(articles['text'])
    similarities = []
    for i in range(len(essays['article_id'].values)):
        for j in articles['article_id'].values:
            if essays['article_id'][i] == j:
                article_embed = article_embeddings[articles.index[articles['article_id']==j].tolist()[0]]
                sim = cosine_similarity(np.asarray(article_embed), np.asarray(essay_embeddings[i]))
                similarities.append(sim.squeeze())
    return similarities


def tokenization(texts, MAX_LENGTH=128, model_name='roberta-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = []
    attention_masks = []

    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=MAX_LENGTH,
            pad_to_max_length='max_length',
            return_attention_mask=True,
            return_tensors='pt' # Return pytorch tensors
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    # convert the list to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks


def convert_labels(raw_labels):
    return torch.tensor(raw_labels.codes, dtype=torch.long)


def data_loader(input_ids, attention_masks, labels, batch_size, shuffle=True):
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(dataset, batch_size = batch_size,shuffle=shuffle)
    return dataloader


def flat_accuracy(preds, labels):
  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()
  count = 0
  for pred, label in zip(pred_flat, labels_flat):
      if pred == label:
        count += 1
  return count / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = round(elapsed, 2)

    return str(datetime.timedelta(seconds=elapsed_rounded))


def initialize_optimizer(model, lr=2e-5, eps=1e-8):
    return AdamW(model.parameters(), lr=lr, eps=eps)


def scheduler(optimizer, epochs, dataloader):
    total_steps = len(dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return scheduler