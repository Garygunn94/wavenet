import pandas as pd
from random import random
from scikits.audiolab import wavread, wavwrite
from numpy import abs, max
 
# Load in the stereo file

#flow = (list(range(1,10,1)) + list(range(10,1,-1)))*100
#pdata = pd.DataFrame({"a":flow, "b":flow})
#pdata.b = pdata.b.shift(9)
#data = pdata.iloc[10:] * random()  # some noise#

import numpy as np

#def _load_data(data, n_prev = 100):
 #   """
 #   data should be pd.DataFrame()
 #   """

#    docX, docY = [], []
#    for i in range(len(data)-n_prev):
#        docX.append(data.iloc[i:i+n_prev].as_matrix())
#        docY.append(data.iloc[i+n_prev].as_matrix())
#    alsX = np.array(docX)
#    alsY = np.array(docY)

#    return alsX, alsY

#def train_test_split(df, test_size=0.1):
  #  """
  #  This just splits data to training and testing parts
  #  """
X_train, fs1, enc1 = wavread('./train/input/mixed_1.wav')
y_train, fs2, enc2 = wavread('./train/output/target_1.wav')
X_test, fs3, enc3 = wavread('./train/input/mixed_4.wav')
y_test, fs4, enc4 = wavread('./train/output/target_4.wav')

X_train = np.array(X_train).reshape(1, len(X_train), 1)
y_train = np.array(y_train).reshape(1, len(y_train), 1)
X_test = np.array(X_test).reshape(1, len(X_test), 1)
y_test = np.array(y_test).reshape(1, len(y_test), 1)
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM

in_out_neurons = len(y_train)
hidden_neurons = len(y_train)

model = Sequential()
model.add(LSTM(hidden_neurons, input_dim=in_out_neurons, return_sequences=False))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")

model.fit(X_train, y_train, batch_size=1, nb_epoch=10, validation_split=0.05)

predicted = model.predict(X_test)
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

# and maybe plot it
wavwrite(predicted, fsf4, enc4)
#pd.DataFrame(predicted).to_csv("predicted.csv")
#pd.DataFrame(y_test).to_csv("test_data.csv")