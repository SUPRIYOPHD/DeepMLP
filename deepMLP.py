"""
Created on Tue Apr 24 14:39:51 2018

@author: Supriyo Mandal
"""
import json
from operator import itemgetter
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras
from IPython.display import SVG
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot

import warnings
warnings.filterwarnings('ignore')

def harmonic_sum(n):
	sum = 0
	for i in range(1,n+1):
		sum += (1.0/i)
	return sum


#dataset = pd.read_json('trunc1L.json', lines=True)
#print (dataset.head())
#print (len(dataset.reviewerID.unique()), len(dataset.asin.unique()))
data = []
with open('./trunc1L.json','r') as f:
	for line in f:
		data.append(json.loads(line))

# finding the helpfulness score for each review
for l in data:
	if l["helpful"][1]!=0 :
		l["helpfulness_score"] = l["helpful"][0]**2.0/l["helpful"][1]
	else :
		l["helpfulness_score"] = 0
       
# finding the rank of each review
sorted_helpful = sorted(data, key=itemgetter('helpfulness_score'), reverse=True) 

productMap = {}

for l in sorted_helpful:
	if l["asin"] in productMap:

		if productMap[l["asin"]][1] == l["helpfulness_score"]:
			l["rank"]=productMap[l["asin"]][0]
			productMap[l["asin"]][2] += 1

		else:
			productMap[l["asin"]][2] += 1
			l["rank"]=productMap[l["asin"]][2]
			productMap[l["asin"]][0] = productMap[l["asin"]][2]
			productMap[l["asin"]][1] = l["helpfulness_score"]

	else:
		productMap[l["asin"]] = [1,l["helpfulness_score"],1]
		l["rank"]=productMap[l["asin"]][0]

# finding the position of each review
sorted_time = sorted(sorted_helpful, key=itemgetter('unixReviewTime'))

productMap = {}

for l in sorted_time:
	if l["asin"] in productMap:
		productMap[l["asin"]] += 1
		l["count"]=productMap[l["asin"]]

	else:
		productMap[l["asin"]] = 1
		l["count"]=productMap[l["asin"]]

# finding the degree of each review
for l in sorted_time:
	l["degree_topmost"]=(1.0/(l["rank"]**2))*(productMap[l["asin"]]-l["count"])
	l["degree_mostrecent"]=harmonic_sum(productMap[l["asin"]]-l["count"])
	l["degree"]=l["degree_topmost"]+l["degree_mostrecent"]
	
    
dataset = pd.DataFrame(data=data)
dataset.reviewerID = dataset.reviewerID.astype('category').cat.codes.values
dataset.asin = dataset.asin.astype('category').cat.codes.values

dataset["helpfulness_score"] = dataset["helpfulness_score"]/dataset["helpfulness_score"].max()
dataset["degree"] = dataset["degree"]/dataset["degree"].max()

train, test = train_test_split(dataset, test_size=0.2)
#print (train.size)
#print (test.size)

# MF implementation in Keras



n_users, n_items = len(dataset.reviewerID.unique()), len(dataset.asin.unique())
n_latent_factors_user = 5
n_latent_factors_item = 8

item_input = keras.layers.Input(shape=[1],name='Item')
item_embedding = keras.layers.Embedding(n_items + 1, n_latent_factors_item, name='Item-Embedding')(item_input)
item_vec = keras.layers.Flatten(name='FlattenItems')(item_embedding)
item_vec = keras.layers.Dropout(0.2)(item_vec)


user_input = keras.layers.Input(shape=[1],name='User')
user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors_user,name='User-Embedding')(user_input))
user_vec = keras.layers.Dropout(0.2)(user_vec)

help_input = keras.layers.Input(shape=[1],name='Helpfulness')
degree_input = keras.layers.Input(shape=[1],name='Degree')

print(user_vec)
concat = keras.layers.concatenate([item_vec, user_vec,help_input])
concat_dropout = keras.layers.Dropout(0.2)(concat)
dense = keras.layers.Dense(200,name='FullyConnected')(concat)
dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(concat)
dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
dense_4 = keras.layers.Dense(25,name='FullyConnected-3', activation='relu')(dense_3)

result = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
adam = Adam(lr=0.01)
model = keras.Model([user_input, item_input,help_input], result)
model.compile(optimizer=adam,loss= 'mean_absolute_error')

#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))

model.summary()

#history1 = model.fit([train.reviewerID, train.asin], train.overall, epochs=5, verbose=1)
#


#train["helpfulness_score"] = train["helpfulness_score"]/train["helpfulness_score"].max()
#train["degree"] = train["degree"]/train["degree"].max()
history1 = model.fit([train.reviewerID, train.asin, train.helpfulness_score], train.overall, epochs=5, verbose=1)
y_hat = np.round(model.predict([test.reviewerID, test.asin, test.helpfulness_score]),0)
y_true = test.overall
print("MSE : ",mean_absolute_error(y_true, y_hat))

#y_hat = np.round(model.predict([test.reviewerID, test.asin]),0)
#y_true = test.helpfulness_score
#print(mean_absolute_error(y_true, y_hat))

#n_latent_factors = 5
#
#item_input = keras.layers.Input(shape=[1],name='Item')
#item_embedding = keras.layers.Embedding(n_items + 1, n_latent_factors, name='Item-Embedding')(item_input)
#item_vec = keras.layers.Flatten(name='FlattenItems')(item_embedding)
#
#user_input = keras.layers.Input(shape=[1],name='User')
#user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(n_users + 1, n_latent_factors,name='User-Embedding')(user_input))
#
#prod = keras.layers.merge([item_vec, user_vec], mode='dot',name='DotProduct')
#model = keras.Model([user_input, item_input], prod)
#model.compile('adam', 'mean_squared_error')
#
#SVG(model_to_dot(model,  show_shapes=True, show_layer_names=True, rankdir='HB').create(prog='dot', format='svg'))
#
