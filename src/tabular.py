# -*- coding: UTF-8 -*-
import torch
from preprocessing import (process_article, read_train, read_dev,
                           concat_change_dtype, combine_articles_to_essays,
                           normalise_numeric)
from score_tools import get_extracted_text
from utils import get_train_dev, initialize_optimizer, scheduler
from trainer import Trainer

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from multimodal_transformers.data import load_data
from transformers import RobertaConfig
from multimodal_transformers.model import RobertaWithTabular
from multimodal_transformers.model import TabularConfig


def data_preparation(raw_training, analysis):
    if analysis == 'whole_article':
        articles = process_article()
        raw_training = combine_articles_to_essays(articles, raw_training)

    if analysis == 'score_based':
        filtered_articles_essays = get_extracted_text()
        raw_training['essay'] = filtered_articles_essays

    return raw_training


def main(analysis='None', problem='empathy', method='tabular', epochs=8, batch_size=8, learning_rate=2e-5,
         max_length=200, model_name='roberta-base', save_path='../models/saved_model.pt'):
    train = read_train()
    dev = read_dev()
    raw_training = concat_change_dtype(train, dev)

    assert analysis == 'whole_article' or analysis == 'score_based' or analysis == 'None'
    assert problem == 'empathy' or problem == 'distress' or problem == 'emotion'

    whole_data = data_preparation(raw_training, analysis)

    numeric = normalise_numeric(whole_data.iloc[:, 8:15])
    whole_data.iloc[:, 8:15] = numeric
    train_whole, dev_whole = get_train_dev(whole_data)

    categorical_cols = ['gender', 'education', 'race']
    numerical_cols = ['age', 'income', 'personality_conscientiousness', 'personality_openess',
                      'personality_extraversion', 'personality_agreeableness', 'personality_stability']

    text_col = ['essay']
    label_col = [problem]

    if problem == 'emotion':
        label_list = ['sadness', 'neutral', 'fear', 'anger', 'disgust', 'surprise', 'joy']
        num_labels = 7
    else:
        label_list = None
        num_labels = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = load_data(
        train_whole,
        text_col,
        tokenizer,
        label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        max_token_length=max_length
    )

    dev_dataset = load_data(
        dev_whole,
        text_col,
        tokenizer,
        label_col,
        label_list=label_list,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        max_token_length=max_length
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True)

    '''configure model'''
    print('Model configuring...')
    roberta_config = RobertaConfig.from_pretrained(model_name)

    tabular_config = TabularConfig(
        combine_feat_method='attention_on_cat_and_numerical_feats',
        cat_feat_dim=15,
        numerical_feat_dim=7,
        num_labels=num_labels,
        use_num_bn=False,
    )

    roberta_config.tabular_config = tabular_config
    model = RobertaWithTabular.from_pretrained(model_name, config=roberta_config)

    device = torch.device("cpu")
    epochs = epochs

    # initialize optimizer and warmup
    optimizer = initialize_optimizer(model, lr=learning_rate)
    warmup = scheduler(optimizer, epochs=epochs, dataloader=train_dataloader)

    trainer = Trainer(method=method, problem=problem)
    trainer.train(train_dataloader, dev_dataloader, model, epochs, optimizer, device, validate=True, warmup=warmup)

    torch.save(model, save_path)


if __name__ == '__main__':
    main(analysis='None', problem='empathy', method='tabular', epochs=1, batch_size=8, learning_rate=2e-5,
         max_length=200, model_name='roberta-base', save_path='../models/test2222T.pt')
