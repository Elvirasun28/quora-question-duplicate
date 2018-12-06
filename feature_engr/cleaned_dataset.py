import numpy as np
import pandas as pd
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation
from tqdm import tqdm
from nltk import word_tokenize
import re
from feature_engr import fea_engr

## load dataset
data_test = pd.read_csv('data/test.csv',index_col=0,header=0,engine='python',encoding='utf-8')
data_train = pd.read_csv('data/train.csv',index_col=0,header=0,engine='python',encoding='utf-8')

print('Clean text beginning ...................................')
cleaned_train = data_train
cleaned_train['question1'] = [fea_engr.cleanText(sent,True,True) for sent in cleaned_train['question1']]
cleaned_train['question2'] = [fea_engr.cleanText(sent,True,True) for sent in cleaned_train['question2'].values]
cleaned_train.to_csv('data/cleaned_train.csv')

cleaned_test = data_test
cleaned_test['question1'] = [fea_engr.cleanText(sent,True,True) for sent in cleaned_test['question1']]
cleaned_test['question2'] = [fea_engr.cleanText(sent,True,True) for sent in cleaned_test['question2']]
cleaned_test.to_csv('data/cleaned_test.csv')