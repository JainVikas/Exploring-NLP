# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 14:17:46 2018

@author: vikas.e.jain
"""

# dataset preparation
from tpot import TPOTClassifier
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import decomposition, ensemble
from sklearn.naive_bayes import MultinomialNB
from sklearn import pipeline

data = pd.read_csv('../dataset/bbc_articles_labels_all.csv')

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['category'])

#Label encoding the target variables
from sklearn.preprocessing import LabelEncoder
categoryLableEncoder = LabelEncoder()
train_y = categoryLableEncoder.fit_transform(train_y)
valid_y = categoryLableEncoder.transform(valid_y)

tfidf_transformer = TfidfVectorizer()
X_train_tfidf = tfidf_transformer.fit_transform(train_x)

X_train_df = pd.DataFrame(X_train_tfidf.toarray())

tpot_clf = TPOTClassifier(generations=10  )
tpot_clf.fit(X_train_df,train_y)