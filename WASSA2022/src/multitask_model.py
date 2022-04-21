# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModel


class MTLModel(torch.nn.Module):
    def __init__(self, n1_out=1, n2_out=1, pretrained_model='roberta-large', drop_rate=0.2):
        super(MTLModel, self).__init__()
        if pretrained_model == 'roberta-large':
            n_in = 1024
        else:
            n_in = 768

        self.pretrained = AutoModel.from_pretrained(pretrained_model)

        self.net1 = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, n1_out))

        self.net2 = nn.Sequential(
            nn.Linear(n_in, 256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, n2_out))

    def forward(self, input_ids, attention_masks):
        outputs = self.pretrained(input_ids, attention_masks)[1]
        return [self.net1(outputs), self.net2(outputs)]


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros(task_num))

    def forward(self, input_ids, attention_masks, targets, problem):

        outputs = self.model(input_ids, attention_masks)

        if problem == 'rgNclf' or problem == 'clfNrg':
            loss1 = torch.sum((targets[0] - outputs[0].squeeze()) ** 2) / targets[0].shape[0]
            loss2 = nn.CrossEntropyLoss()(outputs[1].squeeze(), targets[1]) / targets[1].shape[0]
            loss = loss1 + loss2

        else:
            precision1 = torch.exp(-self.log_vars[0])
            loss = torch.sum(precision1 * (targets[0] - outputs[0].squeeze()) ** 2. + self.log_vars[0], -1)
            precision2 = torch.exp(-self.log_vars[1])
            loss += torch.sum(precision2 * (targets[1] - outputs[1].squeeze()) ** 2. + self.log_vars[1], -1)
            '''
            loss = .0
            for i in range(len(targets)):
                precision = torch.exp(-self.log_vars[i])
                loss += torch.sum(precision * (targets[i] - outputs[i]) ** 2. + self.log_vars[i], -1)
            '''

            loss = torch.mean(loss)

        return loss, outputs, self.log_vars.data.tolist()
