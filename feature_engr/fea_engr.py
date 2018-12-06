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

def cleanText(text,stopword = True,stemming = True):
    try:
        ## delete the special word
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        ## lowercase
        lowercase = text.lower()

        ## delete stopword
        if  stopword:
            stop = set(stopwords.words('english'))
            words = word_tokenize(lowercase)
            stopword_handled = [w for w in words if not w in stop]
        else:
            stopword_handled = lowercase

        stopword_handled = ' '.join(stopword_handled)

        ## deal with abbreviation
        separate_dic = {
            "what's":"what is",
            "'s":" ",
            "'ve":"have",
            "can't":"can not",
            "n't":"not",
            "i'm":"i am",
            "'d":"would",
            "'re":"are",
            "'ll":"will",
        }
        separate_pattern = re.compile(r'\b(' + '|'.join(separate_dic.keys()) + r')\b')
        separated_handled = separate_pattern.sub(lambda x: separate_dic[x.group()], stopword_handled)

        ## other special cases (when second try)
        specialcase_handled = text = re.sub(r"(\d+)(k)", r"\g<1>000", separated_handled)
        specialcase_handled = re.sub(r" e g ", " eg ", specialcase_handled)
        specialcase_handled = re.sub(r" b g ", " bg ", specialcase_handled)
        specialcase_handled = re.sub(r" u s ", " american ", specialcase_handled)
        specialcase_handled = re.sub(r"\0s", "0", specialcase_handled)
        specialcase_handled = re.sub(r" 9 11 ", "911", specialcase_handled)
        specialcase_handled = re.sub(r"e - mail", "email", specialcase_handled)
        specialcase_handled = re.sub(r"j k", "jk", specialcase_handled)
        specialcase_handled = re.sub(r"\s{2,}", " ", specialcase_handled)

        ## delete punctuation
        pun = set(punctuation)
        pun_handlded = ' '.join(w for w in separated_handled.split() if w not in pun)

        ## deal with stem
        if stemming:
            stemmer = SnowballStemmer('english')
            stemmed_handled = [stemmer.stem(word) for word in pun_handlded.split()]
        else:
            stemmed_handled = pun_handlded
        separated_handled = ' '.join(stemmed_handled)
        return separated_handled
    except TypeError:
        print('TypeError: ', text)




