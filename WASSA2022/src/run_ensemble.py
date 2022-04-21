# -*- coding: UTF-8 -*-
from utils import (get_train_dev, get_test_data, tokenization, data_loader,
                   convert_labels, initialize_optimizer, scheduler)
from preprocessing import read_dev, read_train
import pandas as pd
from trainer import Trainer
import torch
from transformers import (RobertaForSequenceClassification, ElectraForSequenceClassification,
                          DebertaForSequenceClassification)
from ensembleDL import ensemble_deep_learning, test_ensembleDL
from tester import write_tsv


def main(mode='train', method='finetune', problem='emotion', batch_size=8, epochs=8, learning_rate=2e-5):
    assert mode == 'train' or mode == 'test'
    if mode == 'train':
        train = read_train()
        dev = read_dev()
        whole = pd.concat([train, dev], axis=0)[['essay', 'emotion']]
        whole['emotion'] = pd.Categorical(whole.emotion)

        train_texts, dev_texts = get_train_dev(whole['essay'])
        train_labels, dev_labels = get_train_dev(whole['emotion'])

        train_labels = convert_labels(train_labels.values)
        dev_labels = convert_labels(dev_labels.values)

        '''tokenization and dataloader'''
        train_idsR, train_masksR = tokenization(train_texts.values, MAX_LENGTH=200, model_name='roberta-base')
        dev_idsR, dev_masksR = tokenization(dev_texts.values, MAX_LENGTH=200, model_name='roberta-base')

        train_dataloaderR = data_loader(train_idsR, train_masksR, train_labels, batch_size=batch_size)
        dev_dataloaderR = data_loader(dev_idsR, dev_masksR, dev_labels, batch_size=batch_size)

        train_idsE, train_masksE = tokenization(train_texts.values, MAX_LENGTH=200,
                                                model_name='google/electra-small-discriminator')
        dev_idsE, dev_masksE = tokenization(dev_texts.values, MAX_LENGTH=200,
                                            model_name='google/electra-small-discriminator')

        train_dataloaderE = data_loader(train_idsE, train_masksE, train_labels, batch_size=batch_size)
        dev_dataloaderE = data_loader(dev_idsE, dev_masksE, dev_labels, batch_size=batch_size)

        train_idsD, train_masksD = tokenization(train_texts.values, MAX_LENGTH=128, model_name='microsoft/deberta-base')
        dev_idsD, dev_masksD = tokenization(dev_texts.values, MAX_LENGTH=128, model_name='microsoft/deberta-base')

        train_dataloaderD = data_loader(train_idsD, train_masksD, train_labels, batch_size=batch_size)
        dev_dataloaderD = data_loader(dev_idsD, dev_masksD, dev_labels, batch_size=batch_size)

        '''Configure models and train'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = epochs
        trainer = Trainer(method=method, problem=problem)

        model1 = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=7,
            output_attentions=False,
            output_hidden_states=False
        )
        optimizer1 = initialize_optimizer(model1, lr=learning_rate)
        warmupSCH1 = scheduler(optimizer1, epochs=epochs, dataloader=train_dataloaderR)

        trainer.train(train_dataloaderR, dev_dataloaderR, model1, epochs, optimizer1, device,
                      validate=False, warmup=warmupSCH1)
        torch.save(model1, '../models/save1.pt')

        model2 = ElectraForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator",
            num_labels=7,
            output_attentions=False,
            output_hidden_states=False
        )
        optimizer2 = initialize_optimizer(model2, lr=learning_rate)
        warmupSCH2 = scheduler(optimizer2, epochs=epochs, dataloader=train_dataloaderE)

        trainer.train(train_dataloaderE, dev_dataloaderE, model2, epochs, optimizer2, device,
                      validate=False, warmup=warmupSCH2)
        torch.save(model2, '../models/save2.pt')

        model3 = DebertaForSequenceClassification.from_pretrained(
            "microsoft/deberta-base",
            num_labels=7,
            output_attentions=False,
            output_hidden_states=False
        )
        optimizer3 = initialize_optimizer(model3, lr=learning_rate)
        warmupSCH3 = scheduler(optimizer3, epochs=epochs, dataloader=train_dataloaderD)

        trainer.train(train_dataloaderD, dev_dataloaderD, model3, epochs, optimizer3, device,
                      validate=False, warmup=warmupSCH3)
        torch.save(model3, '../models/save3.pt')

        del train, dev, whole, train_texts, dev_texts, train_labels, dev_labels, \
            train_idsR, train_masksR, dev_idsR, dev_masksR, \
            train_idsE, train_masksE, dev_idsE, dev_masksE, \
            train_idsD, train_masksD, dev_idsD, dev_masksD, \
            optimizer1, warmupSCH1, optimizer2, warmupSCH2, optimizer3, warmupSCH3

        ensemble_deep_learning(dev_dataloaderR, dev_dataloaderE, dev_dataloaderD, model1, model2, model3)

    else:
        test_data = get_test_data()

        input_ids1, input_mask1 = tokenization(test_data['essay'].values, MAX_LENGTH=200,
                                               model_name='roberta-base')
        input_ids2, input_mask2 = tokenization(test_data['essay'].values, MAX_LENGTH=200,
                                               model_name='google/electra-small-discriminator')
        input_ids3, input_mask3 = tokenization(test_data['essay'].values, MAX_LENGTH=128,
                                               model_name='microsoft/deberta-base')

        print('Start ensemble learning...')

        model1 = torch.load('../models/save1.pt', map_location=torch.device('cpu'))
        model2 = torch.load('../models/save2.pt', map_location=torch.device('cpu'))
        model3 = torch.load('../models/save3.pt', map_location=torch.device('cpu'))
        preds_emo = test_ensembleDL(input_ids1, input_mask1, input_ids2, input_mask2, input_ids3, input_mask3, model1,
                                    model2, model3)
        write_tsv(preds_emo)


if __name__ == '__main__':
    main(mode='train', method='finetune', problem='emotion', batch_size=8, epochs=8, learning_rate=2e-5)
