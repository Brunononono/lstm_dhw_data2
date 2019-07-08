# Import dependencies
import pandas as pd
import numpy as np
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from sklearn.model_selection import train_test_split
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

data = pd.read_csv('features.csv')
labels = pd.read_csv('labels.csv')
model = Sequential()
model.add(LSTM(24,return_sequences=True,input_shape=(3,)))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(data, labels,  epochs=100,  batch_size=1,  verbose=1)