import os
from autogluon.tabular import (
        TabularDataset
        )
from MultilabelPredictor import MultilabelPredictor

# Training Data
labels = ['afm', 'lat', 'sws0', 'B0', 'mag', 'TC']
total_label = ['Formula', 'afm', 'lat', 'sws0', 'B0', 'mag', 'TC']
drop_label = list(set(total_label).difference(set(labels)))
database_name = "data.csv"
_train_data = TabularDataset(database_name)
#columns = ['B0']
#print(_train_data[columns].describe())
#_train_data = TabularDataset(database_name)
#_train_data = _train_data.drop(columns=drop_label)
#
#subsample_size = int(0.8*len(_train_data))
train_data = _train_data.sample(frac=0.8, random_state=42)
train_data.to_csv('train.csv', index=False)
test_index= list(set(_train_data.index.to_list()).difference(set(train_data.index.to_list())))
test_data = _train_data.loc[test_index]
test_data.to_csv('test.csv', index=False)

# Model Initialization
train_data = train_data.drop(columns=drop_label)
test_data = test_data.drop(columns=drop_label)
# labels = ['afm', 'lat', 'sws0', 'B0', 'mag', 'TC']
model_name = "HEA_"
save_path = os.path.join(os.getcwd(),model_name)
#eval_metrics = ['mean_squared_error'] * len(labels)

multi_predictor = MultilabelPredictor(
        labels=labels,
        #eval_metrics=eval_metrics,
        path=save_path
        )
multi_predictor.fit(
        train_data,
        tuning_data=test_data,
        #auto_stack=True,
        #time_limit = None,
        #presets=['good_quality']
        )
