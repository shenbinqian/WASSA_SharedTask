# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import statistics
from statistics import StatisticsError
import torch
import torch.nn.functional as F


def ensemble_deep_learning(dev_dataloader1, dev_dataloader2, dev_dataloader3, model1, model2, model3):
    allPred1 = np.array([])
    allPred2 = np.array([])
    allPred3 = np.array([])
    labels = np.array([])

    for batch in dev_dataloader1:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            result = model1(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        logits = result.logits
        pred1 = np.argmax(F.softmax(logits, dim=-1), axis=-1)
        allPred1 = np.append(allPred1, pred1)

        labels = np.append(labels, b_labels)

    for batch in dev_dataloader2:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            result = model2(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        logits = result.logits
        pred2 = np.argmax(F.softmax(logits, dim=-1), axis=-1)
        allPred2 = np.append(allPred2, pred2)

    for batch in dev_dataloader3:
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_labels = batch[2]

        with torch.no_grad():
            result = model3(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        logits = result.logits
        pred3 = np.argmax(F.softmax(logits, dim=-1), axis=-1)
        allPred3 = np.append(allPred3, pred3)

    print('Max voting...')
    final_pred = np.array([])
    for i in range(len(allPred1)):
        try:
            final_pred = np.append(final_pred, statistics.mode([allPred1[i], allPred2[i], allPred3[i]]))
        except StatisticsError:
            final_pred = np.append(final_pred, pred1[i])

    accuracy = np.mean(final_pred == labels.squeeze())
    print('Accuracy for Emotion prediction is ' + str(accuracy))


def test_ensembleDL(input_ids1, input_mask1, input_ids2, input_mask2, input_ids3, input_mask3, model1, model2, model3):
    allPred1 = np.array([])
    allPred2 = np.array([])
    allPred3 = np.array([])

    with torch.no_grad():
        result1 = model1(input_ids=input_ids1, attention_mask=input_mask1, return_dict=True)
        result2 = model2(input_ids=input_ids2, attention_mask=input_mask2, return_dict=True)
        result3 = model3(input_ids=input_ids3, attention_mask=input_mask3, return_dict=True)

    logits1 = result1.logits
    pred1 = np.argmax(F.softmax(logits1, dim=-1), axis=-1)
    allPred1 = np.append(allPred1, pred1)

    logits2 = result2.logits
    pred2 = np.argmax(F.softmax(logits2, dim=-1), axis=-1)
    allPred2 = np.append(allPred2, pred2)

    logits3 = result3.logits
    pred3 = np.argmax(F.softmax(logits3, dim=-1), axis=-1)
    allPred3 = np.append(allPred3, pred3)

    print('Max voting...')
    final_pred = np.array([])
    for i in range(len(allPred1)):
        try:
            final_pred = np.append(final_pred, statistics.mode([allPred1[i], allPred2[i], allPred3[i]]))
        except StatisticsError:
            final_pred = np.append(final_pred, pred1[i])

    return pd.DataFrame(data={'preds': final_pred.astype(int)})
