import numpy as np
import pandas as pd
from tensorflow.keras import models
import pickle

# load the network
network = models.load_model('model')

# load the network
with open('./adaboost.pkl', 'rb') as f:
    adaboost = pickle.load(f)

# load the data
df_submission = pd.read_pickle('./data/df_submission.pkl.gzip', compression='gzip')


# predict the submission for neural network
X_submit = df_submission[['E','N','Z']]
X_submit = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_submit['E'],X_submit['N'], X_submit['Z']) ],axis=0)

y_pred = network.predict(X_submit)

labels =  y_pred.argmax(axis=1)

df_submit = pd.DataFrame({'trace_id' : df_submission['trace_id'],'submission' : labels})

df_submit.to_csv('./submission_nn.csv',index=False)


# predict the submission for adaboost
X_adaboost = X_submit.reshape(X_submit.shape[0], X_submit.shape[1]*X_submit.shape[2])

y_pred = adaboost.predict(X_adaboost)

df_submit = pd.DataFrame({'trace_id' : df_submission['trace_id'],'submission' : y_pred})

df_submit.to_csv('./submission_adaboost.csv',index=False)


