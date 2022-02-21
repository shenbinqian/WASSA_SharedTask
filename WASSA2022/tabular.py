# -*- coding: UTF-8 -*-
import gc
import torch

from preprocessing import (process_article, read_train, read_dev,
                           concat_change_dtype, combine_articles_to_essays,
                           normalise_numeric)
from utils import get_train_dev, initialize_optimizer, scheduler
from trainer import trainer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from multimodal_transformers.data import load_data
from transformers import RobertaConfig
from multimodal_transformers.model import RobertaWithTabular
from multimodal_transformers.model import TabularConfig

if __name__=='__main__':

    '''first dealing with input data'''
    articles = process_article()
    train = read_train()
    dev = read_dev()
    whole = concat_change_dtype(train, dev)
    whole_data = combine_articles_to_essays(articles, whole)
    numeric = normalise_numeric(whole_data.iloc[:,8:15])

    whole_data.iloc[:, 8:15] = numeric

    train_whole, dev_whole = get_train_dev(whole_data)

    text_col = ['essay']
    label_col = ['emotion']


    categorical_cols = ['gender', 'education', 'race']
    numerical_cols = ['age', 'income', 'personality_conscientiousness', 'personality_openess',
                      'personality_extraversion', 'personality_agreeableness', 'personality_stability']

    label_list = ['sadness', 'neutral', 'fear', 'anger', 'disgust', 'surprise', 'joy']

    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    train_dataset = load_data(
        train_whole,
        text_col,
        tokenizer,
        label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        max_token_length=200
    )

    dev_dataset = load_data(
        dev_whole,
        text_col,
        tokenizer,
        label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        max_token_length=200
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, shuffle=True)

    '''
    # train test split for all related types of data
    train_texts, dev_texts = get_train_dev(whole_data['essay'])

    train_labels, dev_labels = get_train_dev(whole_data['emotion'])
    train_labels = convert_labels(train_labels.values)
    dev_labels = convert_labels(dev_labels.values)

    train_numeric, dev_numeric = get_train_dev(numeric)
    train_numeric = torch.tensor(train_numeric, dtype=torch.float)
    dev_numeric = torch.tensor(dev_numeric, dtype=torch.float)

    # tokenize for text data
    train_ids, train_masks = tokenization(train_texts.values, MAX_LENGTH=300)
    dev_ids, dev_masks = tokenization(dev_texts.values)

    # package all data into dataloader
    train_dataloader = data_loader_numeric(train_ids, train_masks, train_numeric, train_labels, batch_size=8)
    dev_dataloader = data_loader_numeric(dev_ids, dev_masks, dev_numeric, dev_labels, batch_size=8)
    '''
    # delete useless variable for saving RAM
    del articles, train, dev, whole, whole_data, numeric #, train_texts, dev_texts, train_labels, dev_labels, train_numeric, dev_numeric, train_ids, train_masks, dev_ids, dev_masks
    gc.collect()
    print('Input data prepared!')
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    '''configure model'''
    print('Model configuring...')
    roberta_config = RobertaConfig.from_pretrained('roberta-large')

    tabular_config = TabularConfig(
            combine_feat_method='attention_on_cat_and_numerical_feats',
            cat_feat_dim=15,
            numerical_feat_dim=7,
            num_labels=1,
            use_num_bn=False,
    )

    roberta_config.tabular_config = tabular_config

    model = RobertaWithTabular.from_pretrained('roberta-large', config=roberta_config)

    # set number of epochs
    epochs = 8


    #initialize optimizer and warmup
    optimizer = initialize_optimizer(model)
    warmup = scheduler(optimizer, epochs=epochs, dataloader=train_dataloader)


    #train the model
    trainer(train_dataloader, dev_dataloader, model, epochs, optimizer, device, warmup=warmup)