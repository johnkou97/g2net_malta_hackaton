import numpy as np
import pandas as pd
from tensorflow.keras import models

network = models.load_model('model')

# prepare submission

df_submission = pd.read_pickle('./data/df_submission.pkl.gzip', compression='gzip')


X_submit = df_submission[['E','N','Z']]
X_submit = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_submit['E'],X_submit['N'], X_submit['Z']) ],axis=0)


y_pred = network.predict(X_submit)

labels =  y_pred.argmax(axis=1)

df_submit = pd.DataFrame({'trace_id' : df_submission['trace_id'],'submission' : labels})

df_submit.to_csv('./submission.csv',index=False)


