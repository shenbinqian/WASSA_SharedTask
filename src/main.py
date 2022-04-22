# -*- coding: UTF-8 -*-
import pandas as pd
import torch
from transformers import RobertaForSequenceClassification
from preprocessing import read_dev, read_train, process_article, combine_articles_to_essays
from data_aug import clean_data
from score_tools import get_extracted_text
from utils import get_train_dev, tokenization, data_loader, initialize_optimizer, scheduler, weight_embeddings
from trainer import Trainer
from rg_wrapper import RobertaRegressor
import pickle
from multitask_model import MTLModel, MultiTaskLossWrapper
# from embedding import extract_BERT_embeddings
# from RST_parse import extract_nuclei


def get_dataloader(df, batch_size, max_length, model_name, problem):
    train_texts, dev_texts = get_train_dev(df['essay'].values)
    train_ids, train_masks = tokenization(train_texts, MAX_LENGTH=max_length, model_name=model_name)
    dev_ids, dev_masks = tokenization(dev_texts, MAX_LENGTH=max_length, model_name=model_name)
    if problem == 'emotion' or problem == 'classification':
        df['emotion'] = pd.Categorical(df.emotion)
        train_labels, dev_labels = get_train_dev(torch.tensor(df['emotion'].values.codes, dtype=torch.long))
    elif problem == 'empathy':
        train_labels, dev_labels = get_train_dev(df['empathy'].values)
        train_labels = torch.tensor(train_labels, dtype=torch.float)
        dev_labels = torch.tensor(dev_labels, dtype=torch.float)
    else:
        train_labels, dev_labels = get_train_dev(df['distress'].values)
        train_labels = torch.tensor(train_labels, dtype=torch.float)
        dev_labels = torch.tensor(dev_labels, dtype=torch.float)

    train_dataloader = data_loader(train_ids, train_masks, train_labels, batch_size=batch_size)
    dev_dataloader = data_loader(dev_ids, dev_masks, dev_labels, batch_size=batch_size)
    return train_dataloader, dev_dataloader


def data_preparation(raw_training, analysis, batch_size, max_length, model_name, method, problem):
    if analysis == 'whole_article':
        articles = process_article()
        whole = combine_articles_to_essays(articles, raw_training)

        train_dataloader, dev_dataloader = get_dataloader(whole, batch_size, max_length, model_name, problem)

    elif analysis == 'RST_parsing':
        '''
        nuclei = extract_nuclei(directory='../RST-output', check_alignment=True)
        whole_embeddings = extract_BERT_embeddings(raw_training['essay'].values, MAX_LENGTH=200, 
                                                   model_name='roberta-base')
        nuclei_embeddings = extract_BERT_embeddings(nuclei, MAX_LENGTH=200, model_name='roberta-base')

        f = open('../data/embeddings.data', 'wb')
        pickle.dump(whole_embeddings, f)
        pickle.dump(nuclei_embeddings, f)
        f.close()
        '''

        assert method != 'MTL', 'Currently we only support RST parsing for fine-tuning!'

        of = open('../data/embeddings.data', 'rb')
        whole_embeddings = pickle.load(of)
        nuclei_embeddings = pickle.load(of)
        of.close()

        concat_embeddings = weight_embeddings(whole_embeddings, nuclei_embeddings, RST_rate=0.3)
        train_embeddings, dev_embeddings = get_train_dev(concat_embeddings)
        if problem == 'emotion' or problem == 'classification':
            raw_training['emotion'] = pd.Categorical(raw_training.emotion)
            train_labels, dev_labels = get_train_dev(
                torch.tensor(raw_training['emotion'].values.codes, dtype=torch.long))
        elif problem == 'empathy':
            train_labels, dev_labels = get_train_dev(raw_training['empathy'].values)
        else:
            train_labels, dev_labels = get_train_dev(raw_training['distress'].values)

        train_dataloader = data_loader(train_embeddings.float(), train_labels, batch_size=batch_size)
        dev_dataloader = data_loader(dev_embeddings.float(), dev_labels, batch_size=batch_size)

    elif analysis == 'score_based':
        filtered_articles_essays = get_extracted_text()
        raw_training['essay'] = filtered_articles_essays

        train_dataloader, dev_dataloader = get_dataloader(raw_training, batch_size, max_length, model_name, problem)

    elif analysis == 'data_aug':
        problem = 'emotion'
        extra = clean_data(sent_length=25)
        extra.rename(columns={"text": "essay", "labels": "emotion"}, inplace=True)

        whole = raw_training[['essay', 'emotion']]
        df = pd.concat([whole, extra], axis=0)

        train_dataloader, dev_dataloader = get_dataloader(df, batch_size, max_length, model_name, problem)

    else:
        if method == 'MTL':
            train_texts, dev_texts = get_train_dev(raw_training['essay'].values)
            raw_training['emotion'] = pd.Categorical(raw_training.emotion)
            emp_train_labels, emp_dev_labels = get_train_dev(raw_training['empathy'].values)
            dis_train_labels, dis_dev_labels = get_train_dev(raw_training['distress'].values)
            emo_train_labels, emo_dev_labels = get_train_dev(raw_training['emotion'].values.codes)
            train_ids, train_masks = tokenization(train_texts, MAX_LENGTH=max_length, model_name=model_name)
            dev_ids, dev_masks = tokenization(dev_texts, MAX_LENGTH=max_length, model_name=model_name)
            if problem == 'rgNrg':
                train_dataloader = data_loader(train_ids, train_masks,
                                               torch.tensor(emp_train_labels),
                                               torch.tensor(dis_train_labels),
                                               batch_size=batch_size)
                dev_dataloader = data_loader(dev_ids, dev_masks,
                                             torch.tensor(emp_dev_labels),
                                             torch.tensor(dis_dev_labels),
                                             batch_size=batch_size)
            else:
                train_dataloader = data_loader(train_ids, train_masks,
                                               torch.tensor(emp_train_labels),
                                               torch.tensor(emo_train_labels, dtype=torch.long),
                                               batch_size=batch_size)
                dev_dataloader = data_loader(dev_ids, dev_masks,
                                             torch.tensor(emp_dev_labels),
                                             torch.tensor(emo_dev_labels, dtype=torch.long),
                                             batch_size=batch_size)
        else:
            train_dataloader, dev_dataloader = get_dataloader(raw_training, batch_size, max_length, model_name, problem)

    return train_dataloader, dev_dataloader


def main(analysis='None', method='finetune', problem='emotion', batch_size=8, epochs=6, learning_rate=2e-5,
         max_length=200, model_name='roberta-base', save_path='../models/saved_model.pt'):

    train = read_train()
    dev = read_dev()
    raw_training = pd.concat([train, dev], axis=0)

    assert analysis == 'whole_article' or analysis == 'RST_parsing' or analysis == 'score_based' \
           or analysis == 'data_aug' or analysis == 'None'
    assert method == 'finetune' or method == 'MTL', \
        'This file only supports fine-tuning and multi-task learning!' \
        ' For tabular and ensemble models, please run tabular.py and run_ensemble.py'

    train_dataloader, dev_dataloader = data_preparation(raw_training, analysis, batch_size,
                                                        max_length, model_name, method, problem)

    if method == 'MTL':
        assert problem == 'rgNrg' or problem == 'rgNclf' or problem == 'clfNrg'
        if problem == 'rgNrg':
            initialised_model = MTLModel(n1_out=1, n2_out=1, pretrained_model=model_name)
        else:
            initialised_model = MTLModel(n1_out=1, n2_out=7, pretrained_model=model_name)
        model = MultiTaskLossWrapper(2, initialised_model)

    else:
        if problem == 'emotion' or problem == 'classification':
            model = RobertaForSequenceClassification.from_pretrained(
                model_name,
                num_labels=7,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            model = RobertaRegressor(model_name=model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs

    optimizer = initialize_optimizer(model, lr=learning_rate)
    warmup = scheduler(optimizer, epochs=epochs, dataloader=train_dataloader)

    trainer = Trainer(method=method, problem=problem)
    trainer.train(train_dataloader, dev_dataloader, model, epochs, optimizer, device, validate=True, warmup=warmup)

    torch.save(model, save_path)


if __name__ == '__main__':
    main(analysis='None', method='MTL', problem='rgNclf', batch_size=2, epochs=1, learning_rate=2e-5,
         max_length=200, model_name='roberta-base', save_path='../models/test1111.pt')
