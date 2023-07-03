import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score

def evaluate( user_submission_file,true_submission = 'data/submissions_true.csv', phase_codename = 'EVALUATE'):
    df_true = pd.read_csv(true_submission)
    df_true['trace_id'] = df_true['trace_id'].apply(lambda x: int(str(x).replace('trace_','')))
    df_user = pd.read_csv(user_submission_file)
    df_user['trace_id'] = df_user['trace_id'].apply(lambda x: int(str(x).replace('trace_','')))
    df_merge = pd.merge(left=df_true, right=df_user, left_on='trace_id', right_on='trace_id', how='left')
    y_true = df_merge['true_label']
    y_pred = df_merge['submission']
    output = {}
    output["result"] = [
        {
            "train_split": {
                "Precision": precision_score(y_true, y_pred, average='weighted'),
                "Recall": recall_score(y_true, y_pred, average='weighted'),
                "f1-Score": f1_score(y_true, y_pred, average='weighted'),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
            }
        }
    ]
    # To display the results in the result file
    output["submission_result"] = output["result"][0]["train_split"]
    return output['result'][0]['train_split']

# evaluate the submissions

print(f'Neural Network: {evaluate("submission_nn.csv")}')
print(f'Adaboost: {evaluate("submission_adaboost.csv")}')