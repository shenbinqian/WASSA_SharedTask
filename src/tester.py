# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def write_tsv(df):
    if len(df.columns) > 2:
        pre = '../ref/'
    else:
        pre = '../res/'
    name = input("Please enter filename: 'goldstandard', 'predictions_EMP' or 'predictions_EMO' ")
    while name != 'goldstandard' and name != 'predictions_EMP' and name != 'predictions_EMO':
        name = input("Please enter again: 'goldstandard', 'predictions_EMP' or 'predictions_EMO' ")

    filename = pre + name + '.tsv'
    df.to_csv(filename, sep="\t", header=False, index=False)


def flatten_list(_2d_list):
    flat_list = []
    for element in _2d_list:
        for item in element:
            flat_list.append(item)
    return flat_list


def test_labelled_data(dev_dataloader, model, problem):
    model.eval()
    preds = []

    for batch in dev_dataloader:
        b_input_ids = batch['input_ids']
        b_input_mask = batch['attention_mask']
        b_input_numeric = batch['numerical_feats']
        b_input_cat = batch['cat_feats']
        b_labels = batch['labels']

        with torch.no_grad():
            _, logits, _ = model(input_ids=b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels,
                                 numerical_feats=b_input_numeric,
                                 cat_feats=b_input_cat)

        if problem == 'classification' or problem == 'emotion':
            pred = [p.item() for p in np.argmax(F.softmax(logits, dim=-1), axis=-1)]

        else:
            pred = [float(p.numpy()[0]) for p in logits]

        preds.append(pred)

    return pd.DataFrame(data={'preds': flatten_list(preds)})


def test_tabular(input_ids, input_mask, model, problem):
    model.eval()
    with torch.no_grad():
        _, logits, _ = model(input_ids=input_ids,
                             token_type_ids=None,
                             attention_mask=input_mask,
                             labels=None,
                             numerical_feats=None,
                             cat_feats=None)

    if problem == 'classification' or problem == 'emotion':
        preds = [p.item() for p in np.argmax(F.softmax(logits, dim=-1), axis=-1)]

    else:
        preds = [float(p.numpy()[0]) for p in logits]

    return pd.DataFrame(data={'preds': preds})


def test_DLrg(input_ids, input_mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=input_mask)

    preds = [float(p.numpy()[0]) for p in logits]

    return pd.DataFrame(data={'preds': preds})


def test_Embeddings(embeddings, model, problem):
    model.eval()
    with torch.no_grad():
        result = model(inputs_embeds=embeddings, return_dict=True)
    logits = result.logits

    if problem == 'classification' or problem == 'emotion':
        preds = [p.item() for p in np.argmax(F.softmax(logits, dim=-1), axis=-1)]

    else:
        preds = [float(p.numpy()[0]) for p in logits]

    return pd.DataFrame(data={'preds': preds})


def test_MTL(input_ids, input_mask, model, targets, problem):
    model.eval()
    with torch.no_grad():
        _, outputs, _ = model(input_ids, input_mask, targets, problem)

    if problem == 'rgNclf' or problem == 'clfNrg':
        preds_EMP = [float(p.numpy()[0]) for p in outputs[0]]
        preds_EMO = [p.item() for p in np.argmax(F.softmax(outputs[1], dim=-1), axis=-1)]

        return pd.DataFrame(data={'preds': preds_EMP}), pd.DataFrame(data={'preds': preds_EMO})

    else:
        preds1 = [float(p.numpy()[0]) for p in outputs[0]]
        preds2 = [float(p.numpy()[0]) for p in outputs[1]]

        return pd.DataFrame(data={'empathy': preds1, 'distress': preds2})


def test(input_ids, input_mask, model, problem):
    model.eval()
    with torch.no_grad():
        result = model(input_ids=input_ids, attention_mask=input_mask)
    logits = result.logits

    if problem == 'classification' or problem == 'emotion':
        preds = [p.item() for p in np.argmax(F.softmax(logits, dim=-1), axis=-1)]

    else:
        preds = [float(p.numpy()[0]) for p in logits]

    return pd.DataFrame(data={'preds': preds})
