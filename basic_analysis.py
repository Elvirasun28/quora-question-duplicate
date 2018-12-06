import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim


train = pd.read_csv('data/cleaned_train.csv')
test = pd.read_csv('data/cleaned_test.csv')

## basic feature
def basic_analysis(data):
    data['len_q1'] = train['question1'].apply(lambda x: len(str(x)))
    data['len_q2'] = train['question2'].apply(lambda x: len(str(x)))
    data['diff_len'] = train.len_q1 - train.len_q2
    data['common_words'] = train.apply(lambda x: len(set(str(x['question1']).split()).intersection(set(str(x['question2']).split()))), axis=1)

    pre_trained_w2v = gensim.models.KeyedVectors.load_word2vec_format('pre_trained_w2v/GoogleNews-vectors-negative300.bin.gz',binary=True)
    ## Word Mover Distance
    data['wmd'] = data.apply(lambda x: pre_trained_w2v.wmdistance(x[''],x['']),axis = 1)




