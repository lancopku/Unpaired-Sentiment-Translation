import codecs
import json
import glob
from nltk.tokenize import sent_tokenize
import sklearn


def write_file(word_fre):
    writer = codecs.open("emotional_word_fre.txt",'w','utf-8')
    for word in word_fre:
        writer.write(word  + " "+ str(word_fre[word]) +"\n")
    writer.close()






word_fre = dict()
gold_all = []
predicted_all = []
reader = codecs.open('10000sentiment.txt', 'r', 'utf-8')
while True:
    string_ = reader.readline()
    if not string_: break
    dict_example = json.loads(string_)
    review = dict_example["example"]
    gold = dict_example["true-gold"]
    predicted = dict_example["predicted"]
    review_length = len(review.split())
    if review_length > len(gold):
      continue
    gold = gold[:review_length]
    gold_all += gold
    predicted = predicted[:review_length]
    predicted_all += predicted
    for i, word in enumerate(review.split()):
        if word in word_fre:
            word_fre[word] += 1-int(predicted[i])
        else:
            word_fre[word] = 1-int(predicted[i])



print ("f1-score:", sklearn.metrics.f1_score(gold_all, predicted_all))
write_file(word_fre)


