# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/10/11 15:26:48
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   None
'''
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")


def load_data(path):
    """
    数据的每一行代表一个样例和对应的标签，label + "\t" + sample
    """
    samples = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            label, sample = line.strip().split("\t")
            samples.append(sample.strip())
            labels.append(int(label))
    
    return samples, labels


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
