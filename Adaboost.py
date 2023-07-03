# train with Adaboost 
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from MachineLearningUtils import plot_confustion_matrix
import pickle

# label encoding
from sklearn.preprocessing import LabelBinarizer

# load the data
df = pd.read_pickle('./data/df_train.pkl.gzip', compression = 'gzip')
df_test = pd.read_pickle('./data/df_test.pkl.gzip', compression = 'gzip')

X_train = df[['E','N','Z']]
X_test = df_test[['E','N','Z']]
y_train = df['target']
y_test = df_test['target']

X_train = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_train['E'],X_train['N'], X_train['Z']) ],axis=0)
X_test  = np.stack([np.stack([e,n,z], axis=1) for e,n,z in zip(X_test['E'],X_test['N'], X_test['Z']) ],axis=0)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])


# # train the network
n_estimators = 50
# base_estimator = GradientBoostingClassifier()
# adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators)
adaboost = AdaBoostClassifier(n_estimators=n_estimators)
adaboost.fit(X_train, y_train)

# save the network
with open('./adaboost.pkl', 'wb') as f:
    pickle.dump(adaboost, f)

# evaluate the network
y_pred_test = adaboost.predict(X_test)

plot_confustion_matrix(y_test, y_pred_test, df, prob=False)



