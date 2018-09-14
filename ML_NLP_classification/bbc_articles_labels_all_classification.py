# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 11:48:43 2018

@author: vikas.e.jain
"""
# dataset preparation
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

#let's build pipeline using countVectorize (to create bag of word), Tfidf, and a classifier)
text_clf = pipeline.Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

text_clf = text_clf.fit(train_x, train_y)
predicted = text_clf.predict(valid_x)
print(np.mean(predicted == valid_y))
#the result acheived is quite good. however we have used default parameteres, Let's identify the best parameter using randomsearchCV
#In order to use that, we will first need to create a param_dist dict 

#Using grid search for parameter tuning and cross-validation
parameters = {'vect__ngram_range': [(1, 3)],     #unigram-bigram-trigram
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1','l2'),   #To be deeped dive for future purpose
              'clf__alpha': (1e-1,1e-2)}    #To be deeped dive for future purpose
RS_clf = model_selection.RandomizedSearchCV(text_clf, parameters,n_iter=3,scoring="accuracy", cv=5) 

#Fitting model, This steps take a little bit of time to choose the best model available based on tunning parameter
RS_clf = RS_clf.fit(train_x,train_y)
#let's get the bet parameter and score
print("parameters ", RS_clf.best_params_, "Score ", RS_clf.best_score_)


#all this accuracy even without removing or cleaning, let's see what we can achieve if we start extracting details such 
#1. remove stopword, prepositions, stemming, lemmating
#2. do POS tagging for verb, adjectives and noun,
#3. convert to lowercase
# Laoding libraries -Text Processing
from nltk.tag import pos_tag
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import string

wnl = WordNetLemmatizer() 

def word2tag(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

def preprocessText(text):
    #stopword list
    cleaned_sentences=[]
    stopword_list = stopwords.words('english')
    sentences= nltk.sent_tokenize(text)
    for sentence in sentences:
        words= nltk.word_tokenize(sentence)
        tag_words = pos_tag(words)
        cleaned_sentence = " ".join([wnl.lemmatize(tag_word[0], pos=word2tag(tag_word[1])) for tag_word in tag_words if tag_word[0] not in stopword_list])
        cleaned_sentences.append(cleaned_sentence)    
    return ''.join(cleaned_sentences)

#Now let's process the complete data and proceed with remaining steps

data['text'] = data['text'].apply(preprocessText)

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(data['text'], data['category'])

#Label encoding the target variables
from sklearn.preprocessing import LabelEncoder
categoryLableEncoder = LabelEncoder()
train_y = categoryLableEncoder.fit_transform(train_y)
valid_y = categoryLableEncoder.transform(valid_y)

#let's build pipeline using countVectorize (to create bag of word), Tfidf, and a classifier)
text_clf = pipeline.Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),])

text_clf = text_clf.fit(train_x, train_y)
predicted = text_clf.predict(valid_x)
print(np.mean(predicted == valid_y))

parameters = {'vect__ngram_range': [(1, 3)],     #unigram-bigram-trigram
              'tfidf__use_idf': (True, False),
              'tfidf__norm': ('l1','l2'),   #To be deeped dive for future purpose
              'clf__alpha': (1e-1,1e-2)}    #To be deeped dive for future purpose
RS_clf = model_selection.RandomizedSearchCV(text_clf, parameters,n_iter=3,scoring="accuracy", cv=5) 

#Fitting model, This steps take a little bit of time to choose the best model available based on tunning parameter
RS_clf = RS_clf.fit(train_x,train_y)
#let's get the bet parameter and score
print("parameters ", RS_clf.best_params_, "Score ", RS_clf.best_score_)
