# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 15:09:20 2018

@author: vikas.e.jain
"""

#Visualize embddings
from sklearn.manifold import TSNE
import plotly.graph_objs as go
word_embds = model.layers[0].get_weights()

word_list = []
for word, i in tokenizer.word_index.items():
    word_list.append(word)
    
    
X_embedded = TSNE(n_components=2).fit_transform(word_embds)
number_of_words = 1000
trace = go.Scatter(
    x = X_embedded[0:number_of_words,0], 
    y = X_embedded[0:number_of_words, 1],
    mode = 'markers',
    text= word_list[0:number_of_words]
)
layout = dict(title= 't-SNE 1 vs t-SNE 2 for first 1000 words ',
              yaxis = dict(title='t-SNE 2'),
              xaxis = dict(title='t-SNE 1'),
              hovermode= 'closest')
fig = dict(data = [trace], layout= layout)
