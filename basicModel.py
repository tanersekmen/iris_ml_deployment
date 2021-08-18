import numpy as np
import pandas as pd

iris = pd.read_csv('iris.csv')

X = iris.drop('species', axis=1)
y = iris['species']

from sklearn.preprocessing import LabelBinarizer
## there are 3 features for target variables so we encode with LabelBinarizer.
encoder = LabelBinarizer()
y = encoder.fit_transform(y)

## splitting data as train and test
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
## scale the data 
scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_test = scaler.transform(X_test)

## create a model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
scaled_X = scaler.fit_transform(X)
model = Sequential()
model.add(Dense(units=4, activation='relu'))

## last layer for multi-class classification of 3 of them. 
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        

model.fit(scaled_X,y,epochs=epochs)
## saving model and pickle to use deployment.
model.save("final_iris_model.h5")
import joblib
joblib.dump(scaler,'iris_scaler.pkl')
