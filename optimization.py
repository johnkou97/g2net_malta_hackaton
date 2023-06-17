import optuna
import optuna.visualization as vis
import json
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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

# Define the objective function for Optuna
def objective(trial):
    # Define the search space for hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-2)
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.7)
    regularization_rate = trial.suggest_loguniform('regularization_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 8, 16, 32, 64])

    # Define the model architecture using the hyperparameters
    c = 32 #number of channerls per conv layer
    k_size = 3 #size of the convolution kernel
    depth = 7

    network = models.Sequential()
    network.add(Conv1D(c, k_size, activation='relu', strides=2, padding='SAME', input_shape=(6000,3), kernel_regularizer=l2(regularization_rate)))
    for lay in range(10):
        network.add(Conv1D(c, k_size, activation='relu', strides=2, padding='SAME', kernel_regularizer=l2(regularization_rate)))
        
    network.add(Flatten())
    network.add(Dropout(rate=dropout_rate))
    network.add(Dense(48, activation='relu', kernel_regularizer=l2(regularization_rate)))
    network.add(Dense(7, activation='softmax'))

    network.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'] )

    HP_epochs = 50
    HP_ES_patience = 5
    HP_RLR_patience = 2
    HP_batch_size = 2

    cnn_net = network.fit(x=X_train,
                        y=y_train,
                        epochs=HP_epochs,
                        verbose=0,
                        batch_size=HP_batch_size,
                        validation_data=(X_test, y_test),
                        callbacks=[
                            EarlyStopping(monitor='accuracy', patience=HP_ES_patience),
                            ReduceLROnPlateau(verbose=0, patience=HP_RLR_patience, monitor='accuracy'),
                            ModelCheckpoint('best-weights.h5', monitor='accuracy', save_best_only=True, save_weights_only=True)
                        ])

    model.load_weights('best-weights.h5')
    # Evaluate the model on the validation set
    loss, accuracy = network.evaluate(X_test, y_test)

    return accuracy



# Create an Optuna study and optimize the hyperparameters
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Get the best hyperparameters from the study
best_params = study.best_params

with open('best_hyperparameters.json', 'w') as file:
    json.dump(best_params, file)

# Plot the optimization history
fig = vis.plot_optimization_history(study)
fig.write_image('hyperparameter_plots/optimization_history.png', scale=3)

# Plot the parameter importances
fig = vis.plot_param_importances(study)
fig.write_image('hyperparameter_plots/parameter_importances.png', scale=3)

# Plot the parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
fig.write_image('hyperparameter_plots/parallel_coordinate_plot.png', scale=3)

# Plot the parameter importances
fig = vis.plot_edf(study)
fig.write_image('hyperparameter_plots/edf.png', scale=3)

# Plot the parallel coordinate plot
fig = vis.plot_slice(study)
fig.write_image('hyperparameter_plots/slice.png', scale=3)
