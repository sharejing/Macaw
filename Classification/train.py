# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/10/11 15:31:21
@Author  :   Sharejing
@Contact :   yymmjing@gmail.com
@Desc    :   None
'''

from utils import load_data, compute_metrics
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from dataset import Dataset


train_samples, train_labels = load_data("./data/PDDTS_Priority/train_classification.txt")
test_samples, test_labels = load_data("./data/PDDTS_Priority/test_classification.txt")

tokenizer = DistilBertTokenizerFast.from_pretrained("./models/domain-distilbert-base-cased")

train_encodings = tokenizer(train_samples, truncation=True, padding=True)
test_encodings = tokenizer(test_samples, truncation=True, padding=True)

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=5,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True
)

model = DistilBertForSequenceClassification.from_pretrained("./models/domain-distilbert-base-cased", num_labels=4)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics      # evaluation dataset
)

trainer.train()
