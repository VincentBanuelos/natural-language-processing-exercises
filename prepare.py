import pandas as pd
import numpy as np

import re
import unicodedata
import nltk
from nltk.corpus import stopwords

import acquire


def basic_clean(string):
    
    string = string.lower()
    string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode('utf-8')
    string = re.sub(r'[^a-z0-9\'\s]', '', string)
    
    return string

def tokenize(string):
    
    tokenize = nltk.tokenize.ToktokTokenizer()
    string = tokenize.tokenize(string, return_str=True)
    
    return string

def stem(string):
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in string.split()]
    string = ' '.join(stems)
    
    return string

def lemmatize(string):
    
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in string.split()]
    string = ' '.join(lemmas)

    return string

def remove_stopwords(string, extra_words=[], exclude_words=[]):
    stopwords_ls = stopwords.words('english')
    
    stopwords_ls = set(stopwords_ls) - set(exclude_words)
    stopwords_ls = stopwords_ls.union(set(extra_words))
    
    words = string.split()
    filtered_words = [word for word in words if word not in stopwords_ls]
    string = ' '.join(filtered_words)
    
    return string

def clean_df(df, extra_words=[], exclude_words=[]):
    df = df[['title','original']]
    
    df['clean'] = df.original\
                        .apply(basic_clean)\
                        .apply(tokenize)\
                        .apply(remove_stopwords, 
                                    extra_words=extra_words,
                                    exclude_words=exclude_words)
    df['stemmed'] = df.clean.apply(stem)
    df['lemmatized'] = df.clean.apply(lemmatize)
    
    return df