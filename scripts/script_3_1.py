import keras
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.metrics import roc_auc_score
from keras.callbacks import LearningRateScheduler
import math

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def gene_weights_naive(model):
#This function computes the influence of each gene on the model, following the formula (2)
    first_layer_weights = abs(pd.DataFrame(model.get_weights()[0], index = X_test_df.columns))
    second_layer_weights = abs(pd.DataFrame(model.get_weights()[2]))
    output_layers_wieghts = abs(pd.DataFrame(model.get_weights()[3]))
    overall_mean_weight = pd.Series(first_layer_weights.dot(second_layer_weights).dot(output_layers_wieghts).mean(axis=1))
    return(overall_mean_weight)

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

###############################################################################################################
n = X_train_df.shape[1]
m = Y_test_df.shape[1]
r = (n / m) ** (1 / 3)
first_layer = m * (r ** 2)
second_layer = m * r
#Construct model with 20% dropout and l2 kernel regularization
input = keras.layers.Input(shape=(np.shape(X_train_df)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(input)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu", kernel_regularizer=regularizers.l2(0.01))(dropout1)
dropout2 = keras.layers.Dropout(0.2)(hidden1)
output = keras.layers.Dense(Y_train_df.shape[1],activation='softmax')(dropout2)
model_pyramid = keras.models.Model(inputs=[input], outputs=[output])
model_pyramid.summary()
model_pyramid.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
model_pyramid.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Train/Fit
history_pyramid = model_pyramid.fit(X_train, Y_train_df, epochs=10, batch_size=128, validation_split=0.2,callbacks=callbacks_list)
history_df_pyramid = pd.DataFrame(history_pyramid.history)

predicted_prob = model_dropoutL2.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train_df.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Y_test_df.index)
true_type = Y_test_df.idxmax(axis=1)
confusion_model5 = pd.crosstab(predicted_type,true_type)
probs_model5 = pd.DataFrame(predicted_prob, columns = Y_train_df.columns.values, index = Y_test_df.index)

###############################################################################################################


weights = pd.DataFrame(index = X_train_df.columns.values,
                       columns = ['Nothing', 'Dropout Only','Dropout + l1',
                                  'Dropout + l2', 'Pyramid'])
weights['Nothing'] = gene_weights_naive(model_nothing)
weights['Dropout Only'] = gene_weights_naive(model_dropoutonly)
weights['Dropout + l1'] =  gene_weights_naive(model_dropoutL1)
weights['Dropout + l2'] = gene_weights_naive(model_dropoutL2)
weights['Pyramid'] = gene_weights_naive(model_pyramid)
#Divide each column by it's total to get them on same scale for comparison
weights = weights.div(weights.sum(axis=0))
corr = weights.corr() #Correlation matrix of each of the four methods
print(corr)

axes = scatter_matrix(weights, alpha=0.5, diagonal='kde')
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].annotate("%.3f" %corr.values[i,j], (0.8, 0.8), xycoords='axes fraction', ha='center', va='center')
plt.show()

roc_micro = [roc_auc_score(Y_test_df,probs_model1,'micro'),
             roc_auc_score(Y_test_df,probs_model2,'micro'),
             roc_auc_score(Y_test_df,probs_model3,'micro'),
             roc_auc_score(Y_test_df,probs_model4,'micro'),
             roc_auc_score(Y_test_df,probs_model5,'micro')]

roc_macro = [roc_auc_score(Y_test_df,probs_model1,'macro'),
            roc_auc_score(Y_test_df,probs_model2,'macro'),
            roc_auc_score(Y_test_df,probs_model3,'macro'),
            roc_auc_score(Y_test_df,probs_model4,'macro'),
            roc_auc_score(Y_test_df,probs_model5,'macro')]

roc = pd.DataFrame(columns = Y_train_df.columns.values)
roc.loc['Nothing',:] = roc_auc_score(Y_test_df,probs_model1,None)
roc.loc['Dropout Only',:] = roc_auc_score(Y_test_df,probs_model2,None)
roc.loc['Dropout + l1',:] = roc_auc_score(Y_test_df,probs_model3,None)
roc.loc['Dropout + l2',:] = roc_auc_score(Y_test_df,probs_model4,None)
roc.loc['Pyramid',:] = roc_auc_score(Y_test_df,probs_model5,None)
roc['Micro'] = roc_micro
roc['Macro'] = roc_macro