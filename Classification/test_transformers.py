from transformers import pipeline
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import time


tokenizer = DistilBertTokenizerFast.from_pretrained("./model/distilbert-base-cased")
model = DistilBertForSequenceClassification.from_pretrained("./model/checkpoint-195")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


for _ in range(10):
    sentence = input("请输入测试样例：")
    start = time.time()
    test = classifier(sentence)
    end = time.time()
    print(test)
    print("所用时间：", end-start)