# -*- coding: UTF-8 -*-

import torch.nn as nn
from transformers import RobertaModel


class RobertaRegressor(nn.Module):
    def __init__(self, model_name='roberta-base', drop_rate=0.2, freeze=False):
        super(RobertaRegressor, self).__init__()
        if model_name == 'roberta-base':
            D_in, D_out = 768, 1
        else:
            D_in, D_out = 1024, 1

        self.roberta = RobertaModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )
        self.float()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs
