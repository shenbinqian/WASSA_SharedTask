# -*- coding: UTF-8 -*-

import torch.nn as nn
from transformers import RobertaModel

class RobertaRegressor(nn.Module):
    def __init__(self, drop_rate=0.2, freeze=False):
        super(RobertaRegressor, self).__init__()
        D_in, D_out = 1024, 1

        self.roberta = RobertaModel.from_pretrained('roberta-large')
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out)
        )
        self.double()

    def forward(self, input_ids, attention_masks):
        outputs = self.roberta(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs