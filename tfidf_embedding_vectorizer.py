from __future__ import print_function
import json
import re
import csv
import numpy as np
import nltk
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
import sklearn.preprocessing
import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras.backend as K
from collections import defaultdict
from tfidf_embedding_vectorizer import *

# taken from http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = 128

    # each row in X is an array of words
    def fit(self, X):
        vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = nltk.word_tokenize, preprocessor = None, stop_words = stopwords.words('english'), max_features = 10000, lowercase = True)
        vectorizer.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(vectorizer.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, vectorizer.idf_[i]) for w, i in vectorizer.vocabulary_.items()])

        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    # each row in X is an array of words
    def transform(self, X):
        X = [sentence.split() for sentence in X]
        return np.array([np.mean([self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])