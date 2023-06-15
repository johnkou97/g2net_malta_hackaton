# g2net Hackathon in Malta

My attempt at g2net's hackathon in Malta 2020.

## Results
The results are shown on the submission dataset.

|Metric|Score|
|:------:|:-------|
| Precission | 79.5 % |  
| Recall | 79.6 % |
| F1-score | 79.5 % |
| Balanced accuracy | 79.6 % |

## Information on the Hackathon

The data used for the challenge contains earthquakes, together with non-earthquake ambient noise. The goal of this challenge is to correctly classify an unknown set of data `df_submission.pkl.gzip` using a neural network that was trained on `df_train.pkl.gzip` and `df_test.pkl.gzip` where the correct labels are provided. 

The tutorial for the hackathon is [here](https://github.com/zerafachris/g2net_malta_hackaton)

The full set of notebooks and tutorials can be found [here](https://github.com/zerafachris/g2net_2nd_training_school_malta_mar_2020)

You need to download the training, testing and submission datasets from [kaggle](https://www.kaggle.com/datasets/zerafachris/g2net-training-school-hackaton)

To evaluate the model you can get the true submission labels [here](https://github.com/zerafachris/g2net_2nd_training_school_malta_mar_2020/blob/master/HACK_leader_board/Metric_Calculation/data/submissions_true.csv)

