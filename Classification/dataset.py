# -*- encoding: utf-8 -*-
'''
@File    :   datasets.py
@Time    :   2021/10/11 15:53:37
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   None
'''

import torch
import torch.utils.data

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)