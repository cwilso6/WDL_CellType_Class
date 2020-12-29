import keras
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras import initializers
from keras import layers
from keras import regularizers
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import pandas as pd
import os
import collections
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import array
import dill
import math
from itertools import chain
import random as rn
import tensorflow as tf
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pandas.core.common import flatten
import time
import math
from sklearn.metrics import roc_auc_score
from keras.callbacks import LearningRateScheduler
from keras.optimizers import adam

#Importing Chang data and separiating into feature matrix (X) and one hot enconded responses (Y)
Chang = pd.read_table("data/Chang_small.txt", sep = ' ', index_col = 0)
cell_type = Chang['CellType']
Y = pd.get_dummies(cell_type)
X = Chang.drop(columns = 'CellType')

#Split data into training and test sets
X_train_df, X_test_df, Y_train_df, Y_test_df = train_test_split(X, Y)

#Transform the featuares so that they are
#and then apply this to test based on training summary statistics
#z-transform
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_df.values)
X_test = scaler.transform(X_test_df.values)

#Setting up arcitecture for hidden layers (number of nodes)
first_layer = 100
second_layer = 75
#################################################################################
#Construct model with out regularization or dropout
input = keras.layers.Input(shape=(np.shape(X_train_df)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu")(input)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu")(hidden1)
output = keras.layers.Dense(Y_train_df.shape[1],activation='softmax')(hidden2)
model_nothing = keras.models.Model(inputs=[input], outputs=[output])
model_nothing.summary()
model_nothing.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train/Fit
history_nothing = model_nothing.fit(X_train, Y_train_df, epochs=10, batch_size=128, validation_split=0.2)
history_df_noting = pd.DataFrame(history_nothing.history)

predicted_prob = model_nothing.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train_df.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Y_test_df.index)
true_type = Y_test_df.idxmax(axis=1)
confusion_model1 = pd.crosstab(predicted_type,true_type)
probs_model1 = pd.DataFrame(predicted_prob, columns = Y_train_df.columns.values, index = Y_test_df.index)
####################################################################################################
#Construct model with 20% dropout
input = keras.layers.Input(shape=(np.shape(X_train_df)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu")(input)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu")(dropout1)
dropout2 = keras.layers.Dropout(0.2)(hidden1)
output = keras.layers.Dense(Y_train_df.shape[1],activation='softmax')(dropout2)
model_dropoutonly = keras.models.Model(inputs=[input], outputs=[output])
model_dropoutonly.summary()
model_dropoutonly.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train/Fit
history_dropooutonly = model_dropoutonly.fit(X_train, Y_train_df, epochs=10, batch_size=128, validation_split=0.2)
history_df_dropooutonly = pd.DataFrame(history_dropooutonly.history)

predicted_prob = model_dropoutonly.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train_df.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Y_test_df.index)
true_type = Y_test_df.idxmax(axis=1)
confusion_model2 = pd.crosstab(predicted_type,true_type)
probs_model2 = pd.DataFrame(predicted_prob, columns = Y_train_df.columns.values, index = Y_test_df.index)
####################################################################################################
#Construct model with 20% dropout and l1 kernel regularization
input = keras.layers.Input(shape=(np.shape(X_train_df)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu",kernel_regularizer=regularizers.l1(0.01))(input)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu", kernel_regularizer=regularizers.l1(0.01))(dropout1)
dropout2 = keras.layers.Dropout(0.2)(hidden1)
output = keras.layers.Dense(Y_train_df.shape[1],activation='softmax')(dropout2)
model_dropoutL1 = keras.models.Model(inputs=[input], outputs=[output])
model_dropoutL1.summary()
model_dropoutL1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train/Fit
history_dropooutL1 = model_dropoutL1.fit(X_train, Y_train_df, epochs=10, batch_size=128, validation_split=0.2)
history_df_dropooutL1 = pd.DataFrame(history_dropooutL1.history)

predicted_prob = model_dropoutL1.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train_df.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Y_test_df.index)
true_type = Y_test_df.idxmax(axis=1)
confusion_model3 = pd.crosstab(predicted_type,true_type)
probs_model3 = pd.DataFrame(predicted_prob, columns = Y_train_df.columns.values, index = Y_test_df.index)

####################################################################################################
#Construct model with 20% dropout and l2 kernel regularization
input = keras.layers.Input(shape=(np.shape(X_train_df)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(input)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu", kernel_regularizer=regularizers.l2(0.01))(dropout1)
dropout2 = keras.layers.Dropout(0.2)(hidden1)
output = keras.layers.Dense(Y_train_df.shape[1],activation='softmax')(dropout2)
model_dropoutL2 = keras.models.Model(inputs=[input], outputs=[output])
model_dropoutL2.summary()
model_dropoutL2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train/Fit
history_dropooutL2 = model_dropoutL2.fit(X_train, Y_train_df, epochs=10, batch_size=128, validation_split=0.2)
history_df_dropooutL2 = pd.DataFrame(history_dropooutL2.history)

predicted_prob = model_dropoutL2.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train_df.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Y_test_df.index)
true_type = Y_test_df.idxmax(axis=1)
confusion_model4 = pd.crosstab(predicted_type,true_type)
probs_model4 = pd.DataFrame(predicted_prob, columns = Y_train_df.columns.values, index = Y_test_df.index)








