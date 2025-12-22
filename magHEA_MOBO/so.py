import os
from src.bayesian_optimization import BayesianOptimization
import pandas as pd

root = os.getcwd()
os.popen(f'rm -r {root}/performance_record.txt {root}/model_weights')

FI = pd.read_csv('../HEA_Data/Feature_importance.dat')
features = FI['Features'].tolist()[:30]
#print(features)

target_props = ['B0', 'mag']
data_file = 'Quat.csv'
cand_file = 'Quat_cand.csv'
#df = pd.read_csv('../HEA_Data/quat_4magpie_original_HR.csv')
df = pd.read_csv('../HEA_Data/hte_quat_stable.csv')
headers = ['composition'] + features + target_props
df[headers].sample(n=20, random_state=996).to_csv(data_file)
#df[target_props] = None
#df[headers].sample(frac=0.25, random_state=996).to_csv(data_file)
#print(headers)
#print(df[target_props].describe())

model_list = [
        'Ridge',
        'Lasso',
        'ElasticNet',
        'KNeighborsRegressor',
        'DecisionTreeRegressor',
        'RandomForest',
#        'SVR',
#        'MLPRegressor',
#        'GradientBoostingRegressor',
#        'AdaBoostRegressor',
#        'ExtraTreesRegressor',
#        'XGBoost',
#        'LightGBM',
#        'GaussianProcess',
#        'FastKAN'
        ]

BO = BayesianOptimization(
        target_props,
        data_file,
        model_list=model_list,
        acq_method='ucb',
        stacking=True,
        #close_pool_initial_samples=10,
        #close_pool=True,
        )
#BO.close_pooling_test(
#        n_bootstrap_sample_nums=30,
#        n_iter=100,
#        batch_size=10,
#        hpar=1
#        )
BO.optimize(
        batch_size=10,
        n_bootstrap_sample_nums=30,

        )
