import os, pickle
from src.io import IOManager
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_models(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

def model_predict(model, X):
    preds = model.predict(X)
    return preds

def main():
    fn = './model_weights/Lasso_0_bootstrap.pkl'
    data = load_models(fn)
    print(data)
    model_name = data['model_name']
    models = data['models']
    print(model_name, models)
    # io_manager = IOManager()
    # drop_props = ['Cauchy','M_s','T_C','sws0','composition','B0']
    target_props = ['Pugh']
    file_name = '3d_metal_data.csv'
    df = pd.read_csv(file_name)
    feature_props = [col for col in df.columns if col not in ['Cauchy','M_s','T_C','sws0','composition','B0'] and col not in target_props]
    X = df[feature_props].to_numpy()
    y = df[target_props].to_numpy()
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    print(X.shape)
    print(y.ndim)
    scaler_X.fit(X)
    scaler_y.fit(y)
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)
    # y_predict_scaled = model_predict(model, X_scaled)
    predictions = [model_predict(model, X_scaled) for model in models]
    # y_predict_scaled = np.expand_dims(y_predict_scaled, -1)
    predictions = np.array(predictions)
    print(predictions.shape)
    y_predict_scaled = np.mean(predictions, axis=0)
    print(y_predict_scaled.shape)
    print(y_scaled.flatten().shape)
    # y_predict = scaler_y.inverse_transform(y_predict_scaled)
    # print(y_predict)
    r2 = r2_score(y_scaled.flatten(), y_predict_scaled.flatten())
    print("R^2 Accuracy:", r2)
    plt.figure(figsize=(8, 6))

    # Scatter plot of true vs predicted values
    plt.scatter(y_scaled.flatten(), y_predict_scaled.flatten(), color="blue", alpha=0.6, label="Predictions")

    # Add a 1:1 line (perfect prediction reference)
    # min_val = min(0, min(y_pred))
    # max_val = max(max(y_true), max(y_pred))
    # plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")

    # Add labels, title, and legend
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted Values ($R^2$ = {r2:.2f})")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig('test.png')
    plt.show()
    # if y.ndim == 1:
    #     y = np.expand_dims(y, -1)
    # # X, y = io_manager.read_data(file_name, target_props=target_props, feature_props=feature_props)
    # print(X.shape)
    # print(y.shape)
    # X_scaled = io_manager.standardize_data(X=X, y=None)
    # y_predict_scaled = model_predict(model, X_scaled)
    # print(y_predict_scaled.shape)
    # y_predict_scaled = np.expand_dims(y_predict_scaled, -1)
    # print(y_predict_scaled.shape)

    # y_predict = io_manager.inverse_transform_y(y_predict_scaled)
    # print(y_predict)
    return 
if __name__ == "__main__":
    # fn = './model_weights/AdaBoostRegressor_0_bootstrap.pkl'
    # data = load_models(fn)
    # # print(data)
    # model_name = data['model_name']
    # print(model_name)
    # models = data['models']
    # print(models[0])
    # with open('test.pkl', 'wb') as f:
    #     pickle.dump({'model_name': model_name, 'model': models[0]}, f)
    main()