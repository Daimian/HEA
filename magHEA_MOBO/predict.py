import os
from src.bayesian_optimization import BayesianOptimization
import pandas as pd
import logging
import sys

logging.basicConfig(
    filename='bo.log',
    filemode='a',
    format='%(asctime)s\t%(message)s',
    level=logging.INFO
)

target_props = ['M_s', 'T_C', 'Pugh', 'Cauchy']
# target_props = ['Cauchy']
fi_len = -1
# init_len = 10
#fn_data = '3d_metal_data.csv'
fn_data = 'Pareto_data.csv'
# fn_cand = '3d_metal_cand.csv'
fn_todo = '3d_metal_todo.csv'
# raw_data = '../HEA_Data/3d_metal_o5_Smix.csv'
model_list = [
        # 'Ridge',
        # 'Lasso',
        'ElasticNet',
        'KNeighborsRegressor',
        # 'DecisionTreeRegressor',
        # 'RandomForest',
        'SVR',
        'MLPRegressor',
        'GradientBoostingRegressor',
        'AdaBoostRegressor',
        # 'ExtraTreesRegressor',
        'XGBoost',
        # 'LightGBM',
#        'GaussianProcess',
#        'FastKAN'
        ]

def main():
        #if (not os.path.isfile(fn_data)) or (not os.path.isfile(fn_cand)):
        #       init()
        # Loop
        logging.info("Entering the BO loop.")
        # df = pd.read_csv(fn_data)
        # feature_props = list(df.columns) - ['sws0'] - target_props
        # feature_props = [col for col in df.columns if col not in ['sws0','composition','B0'] and col not in target_props]
        # print(feature_props)
        BO = BayesianOptimization(
                target_props,
                fn_data,
                model_list=model_list,
                acq_method='ucb',
                stacking=False,
                # feature_props=feature_props
                )
        BO.optimize(
                n_bootstrap_sample_nums=20,
                batch_size=20,
                n_samples=20,
                iterations=1,
                # num_candidate=300,
                # hpar=0.01
                )
        # samples, next_indices= BO.optimize(
        #         batch_size=10,
        #         n_bootstrap_sample_nums=30,
        #         hpar=0.1
        #         )
        # logging.info(f'Next Index: {next_indices}')
        # logging.info("Saving the todo-list. Removing it from the candidate file.")
        # logging.info(f"The todo-list: {fn_todo[['composition']+target_props]}")
        return 0
if __name__ == '__main__':
        # init()
        main()
        #for i in range(10):
        #    main()
