import keras
from keras import regularizers
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

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
Y_train = pd.get_dummies(cell_type)
X_train = Chang.drop(columns = 'CellType')

#Importing Tirosh data and separiating into feature matrix (X) and one hot enconded responses (Y)
Tirosh = pd.read_table("data/Tirosh_small.txt", sep = ' ', index_col = 0)
cell_type = Chang['CellType']
Y_test = pd.get_dummies(cell_type)
X_test = Chang.drop(columns = 'CellType')
