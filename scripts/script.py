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

Chang = pd.read_table("data/Chang_small.txt", sep = ' ')



