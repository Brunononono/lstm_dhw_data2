# Import dependencies
import pandas as pd
import plotter
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

features = pd.read_csv('features.csv')
labels = pd.read_csv('labels.csv')


x_train, x_test, y_train, y_test = train_test_split(features,labels,test_size=0.2)


model = Sequential()

model.add(LSTM(24,activation='relu',input_shape=(1,3)))

model.add(Dense(1))

model.compile(loss='mse', optimizer="adam")

model.fit(x_train,y_train,epochs=60,verbose=2)

results=model.predict(x_test)

print(results)
