import keras
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def gene_weights_naive(model):
#This function computes the influence of each gene on the model, following the formula (2)
    first_layer_weights = abs(pd.DataFrame(model.get_weights()[0], index = X_test_df.columns))
    second_layer_weights = abs(pd.DataFrame(model.get_weights()[2]))
    output_layers_wieghts = abs(pd.DataFrame(model.get_weights()[3]))
    overall_mean_weight = pd.Series(first_layer_weights.dot(second_layer_weights).dot(output_layers_wieghts).mean(axis=1))
    return(overall_mean_weight)

#Importing Chang data and separiating into feature matrix (X) and one hot enconded responses (Y)
Chang = pd.read_table("data/Chang_small.txt", sep = ' ', index_col = 0)
#Drop cells types that are not in Tirosh
Chang = Chang.loc[Chang['CellType'].str.contains("DCs|Plasma_cells|pDCs|Myofibroblasts|Naive|Tumor") == False]
Chang['CellType'] = Chang['CellType'].astype('category')
#Recode categories to match those in Tirosh
mapper = {'B_cells': 'B_cells','CAFs': 'CAFs','CD4_T_cells': 'CD4_T_cells', 'CD8_act_T_cells': 'CD8_T_cells',
          'CD8_ex_T_cells': 'CD8_T_cells', 'CD8_mem_T_cells': 'CD8_T_cells', 'Macrophages': 'Macrophages',
          'Melanocytes':'Melanocytes', 'Tfh': 'CD4_T_cells'}
Chang['CellType'] = Chang['CellType'].map(mapper).fillna(Chang['CellType'])
cell_type_train = Chang['CellType'].astype('category')

Y_train = pd.get_dummies(Chang['CellType'])
X_train = Chang.drop(columns = 'CellType')

#Importing Tirosh data and separiating into feature matrix (X) and one hot enconded responses (Y)
Tirosh = pd.read_table("data/Tirosh_small.txt", sep = ' ', index_col = 0)
cell_type_test = Tirosh['CellType']

Y_test = pd.get_dummies(cell_type_test)
X_test = Tirosh.drop(columns = 'CellType')

#Z-transformation for the training and test sets, where the mean and variacnes from the
#training set are used to scale the testing data
#This might not be appropriate since were orginally on diffeent scale,
#This will change the correlation stucture in the test set becuase of using
#potentially misspecified means
scaler = StandardScaler()
X_train_val = scaler.fit_transform(X_train.values)
X_test_val = scaler.transform(X_test.values)

#Setting up arcitecture for hidden layers (number of nodes)
first_layer = 100
second_layer = 75

#Naive Model
input = keras.layers.Input(shape=(np.shape(X_train)[1],))
hidden1 = keras.layers.Dense(first_layer, activation="relu",kernel_regularizer=regularizers.l2(0.01))(input)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(second_layer, activation="relu",kernel_regularizer=regularizers.l2(0.01))(dropout1)
dropout2 = keras.layers.Dropout(0.2)(hidden2)
output = keras.layers.Dense(Y_train.shape[1],activation='softmax')(dropout2)
model_naive = keras.models.Model(inputs=[input], outputs=[output])
model_naive.summary()
model_naive.compile(loss='categorical_crossentropy', optimizer="adam" ,metrics=['accuracy'])

#Train/Fit
history_naive = model_naive.fit(X_train_val, Y_train, epochs=10, batch_size=128)
history_naive_df = pd.DataFrame(history_naive.history)

predicted_prob = model_naive.predict(X_test)
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Tirosh.index)
confusion_model = pd.crosstab(predicted_type,Tirosh['CellType'])
probs_model = pd.DataFrame(predicted_prob, columns = Y_train.columns.values, index = Tirosh.index)
probs_model['Truth'] = cell_type_test
probs_model['predicted'] = predicted_type
##############################################################################################################
#WDL
bio_relevant = ['CD4','CD8A', 'CD8B', 'GZMK','MIA', 'PMEL', 'MLANA'] #Presented in the paper
bio_relevant = list(set(bio_relevant).intersection(set(X_train.columns.values))) #Genes in the subset of data
#Now split data into all inputs A and B
X_train_A = X_train[bio_relevant]
X_test_A = X_test[bio_relevant]
X_train_B = X_train.drop(columns = bio_relevant)
X_test_B = X_test.drop(columns = bio_relevant)

#model (two input now, features 0:4 to patah A, 2:7 to path B)
input_A = keras.layers.Input(shape=(np.shape(X_train_A)[1],))
input_B = keras.layers.Input(shape=(np.shape(X_train_B)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(input_B)
#hidden1 = keras.layers.Dense(int(first_layer), activation="relu")(input_B)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
hidden2 = keras.layers.Dense(int(second_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(hidden1)
#hidden2 = keras.layers.Dense(int(second_layer), activation="relu")(hidden1)
dropout2 = keras.layers.Dropout(0.2)(hidden2)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(Y_train.shape[1],activation='softmax')(concat)
model_wdl = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
model_wdl.summary()
model_wdl.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

history_wdl = model_wdl.fit([X_train_A.to_numpy(), X_train_B.to_numpy()], Y_train, epochs=10, batch_size=128)

history_wdl_df = pd.DataFrame(history_wdl.history)

predicted_prob = model_wdl.predict([X_test_A, X_test_B])
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Tirosh.index)
confusion_model = pd.crosstab(predicted_type,Tirosh['CellType'])
probs_wdl_model = pd.DataFrame(predicted_prob, columns = Y_train.columns.values, index = Tirosh.index)
probs_wdl_model['Truth'] = cell_type_test
probs_wdl_model['predicted'] = predicted_type
#########################################################################################################
#Wide component in the first hidden layer
#model (two input now, features 0:4 to patah A, 2:7 to path B)
input_A = keras.layers.Input(shape=(np.shape(X_train_A)[1],))
input_B = keras.layers.Input(shape=(np.shape(X_train_B)[1],))
hidden1 = keras.layers.Dense(int(first_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(input_B)
dropout1 = keras.layers.Dropout(0.2)(hidden1)
concat = keras.layers.concatenate([input_A, hidden1])
hidden2 = keras.layers.Dense(int(second_layer), activation="relu",kernel_regularizer=regularizers.l2(0.01))(concat)
dropout2 = keras.layers.Dropout(0.2)(hidden2)
output = keras.layers.Dense(Y_train.shape[1],activation='softmax')(dropout2)
model_wdl_first = keras.models.Model(inputs=[input_A, input_B], outputs=[output])
model_wdl_first.summary()
model_wdl_first.compile(loss='categorical_crossentropy', optimizer="adam",metrics=['accuracy'])

history_wdl_first = model_wdl_first.fit([X_train_A.to_numpy(), X_train_B.to_numpy()], Y_train, epochs=10, batch_size=128)

history_wdl_first_df = pd.DataFrame(history_wdl_first.history)

predicted_prob = model_wdl_first.predict([X_test_A, X_test_B])
predicted = [np.where(predicted_prob[i] == max(predicted_prob[i]))[0][0] for i in range(predicted_prob.shape[0])]
predicted_type = [Y_train.columns.values[i] for i in predicted]
predicted_type = pd.Series(predicted_type, index = Tirosh.index)
confusion_model = pd.crosstab(predicted_type,Tirosh['CellType'])
probs_wdl_first_model = pd.DataFrame(predicted_prob, columns = Y_train.columns.values, index = Tirosh.index)
probs_wdl_first_model['Truth'] = cell_type_test
probs_wdl_first_model['predicted'] = predicted_type
###################################################################################################
#AUC Comparisons

roc_micro = [roc_auc_score(Y_test,probs_model.drop(columns = ['Truth', 'predicted']),'micro'),
             roc_auc_score(Y_test,probs_wdl_model.drop(columns = ['Truth', 'predicted']),'micro'),
             roc_auc_score(Y_test,probs_wdl_first_model.drop(columns = ['Truth', 'predicted']),'micro')]

roc_macro = [roc_auc_score(Y_test,probs_model.drop(columns = ['Truth', 'predicted']),'macro'),
             roc_auc_score(Y_test,probs_wdl_model.drop(columns = ['Truth', 'predicted']),'macro'),
             roc_auc_score(Y_test,probs_wdl_first_model.drop(columns = ['Truth', 'predicted']),'macro')]

roc = pd.DataFrame(columns = Y_train.columns.values)
roc.loc['Naive',:] = roc_auc_score(Y_test,probs_model.drop(columns = ['Truth', 'predicted']),None)
roc.loc['WDL Second Layer',:] = roc_auc_score(Y_test,probs_wdl_model.drop(columns = ['Truth', 'predicted']),None)
roc.loc['WDL First Layer',:] = roc_auc_score(Y_test,probs_wdl_first_model.drop(columns = ['Truth', 'predicted']),None)
roc['Micro'] = roc_micro
roc['Macro'] = roc_macro