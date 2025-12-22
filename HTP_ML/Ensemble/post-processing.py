from autogluon.tabular import TabularDataset, TabularPredictor
import matplotlib.pyplot as plt
import numpy as np

train_data = TabularDataset('train.csv')
test_data = TabularDataset('test.csv')

columns = ['lat','afm','sws0','B0','mag','TC']
targets = ['accuracy'] + ['r2']*(len(columns)-1)
train_r2 = []
eval_r2 = []
for label,target in zip(columns, targets):
    predictor = TabularPredictor.load(f"HEA_Predictor_{label}")
    # print(predictor.leaderboard())
    train_r2.append(predictor.evaluate(train_data)[target])
    eval_r2.append(predictor.evaluate(test_data)[target])
print(columns)
print(train_r2)
print(eval_r2)

