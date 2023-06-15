import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MachineLearningUtils import plot_confustion_matrix, plot_network_learning_graphs
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# load the data
df = pd.read_pickle('./data/df_train.pkl.gzip', compression = 'gzip')
df_test = pd.read_pickle('./data/df_test.pkl.gzip', compression = 'gzip')

X_train = df[['E','N','Z']]
X_test = df_test[['E','N','Z']]
y_train = df['target']
y_test = df_test['target']

X_train = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_train['E'],X_train['N'], X_train['Z']) ],axis=0)
X_test  = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_test['E'],X_test['N'], X_test['Z']) ],axis=0)

y_train = to_categorical(y_train,7)
y_test = to_categorical(y_test,7)

# define and fit the network
c = 32 #number of channerls per conv layer
k_size = 3 #size of the convolution kernel
depth = 7

network = models.Sequential()
network.add(Conv1D(c, k_size, activation='relu', strides=2, padding='SAME', input_shape=(6000,3)))
for lay in range(10):
    network.add(Conv1D(c, k_size, activation='relu', strides=2, padding='SAME'))
    
network.add(Flatten())
network.add(Dense(48, activation='relu'))
network.add(Dense(7, activation='softmax'))

network.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'] )

HP_epochs = 50
HP_ES_patience = 5
HP_RLR_patience = 2
HP_batch_size = 2

cnn_net = network.fit(x=X_train,
                      y=y_train,
                      epochs=HP_epochs,
                      verbose=1,
                      batch_size=HP_batch_size,
                      validation_data=(X_test, y_test),
                      callbacks=[
                          EarlyStopping(monitor='val_loss', patience=HP_ES_patience),
                          ReduceLROnPlateau(verbose=1, patience=HP_RLR_patience, monitor='val_loss')
                      ])

# save the network
network.save('model')

# generate predictions on testing data
y_pred_test = network.predict(X_test)

# create plots
plot_network_learning_graphs(cnn_net)

plot_confustion_matrix(y_test, y_pred_test, df, prob=True)

