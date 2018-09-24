# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:43:24 2018

@author: vikas.e.jain
"""

"""
file to use classification on bbc article using keras, text embeddign matrix
"""
# dataset preparation

import pandas as pd
import numpy as np

data = pd.read_csv('../../dataset/bbc_articles_labels_all.csv')

texts=data.text.tolist()
category_labels=data.category.tolist()
#since category labels needs to be represented as number for line40 
from sklearn.preprocessing import LabelEncoder
categoryLableEncoder = LabelEncoder()
category_labels = categoryLableEncoder.fit_transform(category_labels)

#setup variables need
MAX_NB_WORDS=1000
MAX_SEQUENCE_LENGTH=1000
VALIDATION_SPLIT = 0.1

## lets preporcess the text and tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(category_labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

#let's prepare an embedding layer
#if the glove is not alreay available, go ahead and download from here:https://worksheets.codalab.org/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/
import os
GLOVE_DIR = "../WordEmbedding/"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


#Let's now leverage that and make an embedding matrix for our scenario
EMBEDDING_DIM =100 #glove has 100 len vectors, so in order to broadcast properly we would need minimum 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
        
#let's load the embedding layer and use this embedding matrix for intialization
#note: the trainable is false, so this layer's weight won't change as we will train our network
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

#Training a LSTM network
#we had earlier imported Embedding layer. let's import the remaining
from keras.models import Sequential
from keras.layers import Dense,LSTM, Activation
from keras.layers.embeddings import Embedding
model = Sequential()
model.add(Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(labels.shape[1], activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# happy learning!
model_train_history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=5, batch_size=64)

