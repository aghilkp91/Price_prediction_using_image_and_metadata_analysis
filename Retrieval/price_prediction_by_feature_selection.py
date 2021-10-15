import argparse
import RetrieveImage as retrieveImage
import os
import csv
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import *
import json
import pandas as pd
import metadata_preprocessing as processing
import xgboost as xgb
import DeepImageUtils as IU
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data'
images_path = "/ceph/csedu-scratch/project/akaradathodi/Images/"
# images_path = os.path.abspath(data_path + '/../../') + '/Images/'

image_features_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data/total_features.npz'

def add_features_to_df(train_df, validation_df):
    #Loading image features and adding it to dataframe
    loaded_features = np.load(image_features_path, allow_pickle=True)
    features = list(loaded_features['arr_0'])
    for feature in features:
        if feature[2] in train_df.uniq_id:
            train_df.loc[train_df.index[train_df['uniq_id'] == feature[2]], 'image_feature'] = feature[1]
            # train_df['image_feature'] = feature[1]
        if feature[2] in validation_df.uniq_id:
            # validation_df['image_feature'] = feature[1]
            validation_df.loc[validation_df.index[validation_df['uniq_id'] == feature[2]], 'image_feature'] = feature[1]
    print('Train DataFrame 0th row')
    print(train_df.iloc[0])
    return train_df, validation_df

def add_features_to_testdf(test_df):
    for index, row in test_df.iterrows():
        img = images_path + row['uniq_id'] + '_0.jpg'
        # print(img)
        feature_vector = IU.CreateImageFeaturesVector(img)
        row['image_feature'] = feature_vector
    print('Test DataFrame 0th row')
    print(test_df.iloc[0])
    return test_df

def run_on_xgb_regressor(train_data, y_train, test_data):
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(train_data, y_train)
    predictions = xgb_model.predict(test_data)
    return predictions

def run_on_ridge_regression(train_data, y_train, test_data):
    ridge_model = Ridge(solver='lsqr', fit_intercept=False)  # solver='lsqr' reduces time to train significantly
    ridge_model.fit(train_data, y_train)

    predictions = ridge_model.predict(test_data)
    return predictions

def run_on_light_bgm_regressor(train_data, y_train, test_data):
    lgbm_model = LGBMRegressor()
    lgbm_model.fit(train_data, y_train)
    predictions = lgbm_model.predict(test_data)
    return predictions

def find_rmse_mae_r2(predicted_price_xgb, predicted_price_ridge, predicted_price_lightbgm, actual_price):
    predicted_price_xgb = np.array(predicted_price_xgb)
    predicted_price_ridge = np.array(predicted_price_ridge)
    predicted_price_lightbgm = np.array(predicted_price_lightbgm)
    actual_price = np.array(actual_price)

    results = {
            'MSE_xgb': mean_squared_error(predicted_price_xgb, actual_price),
            'MAE_xgb': mean_absolute_error(predicted_price_xgb, actual_price),
            'R2_xgb': r2_score(predicted_price_xgb, actual_price),
            'MSE_ridge': mean_squared_error(predicted_price_ridge, actual_price),
            'MAE_ridge': mean_absolute_error(predicted_price_ridge, actual_price),
            'R2_ridge': r2_score(predicted_price_ridge, actual_price),
            'MSE_lightbgm': mean_squared_error(predicted_price_lightbgm, actual_price),
            'MAE_lightbgm': mean_absolute_error(predicted_price_lightbgm, actual_price),
            'R2_lightbgm': r2_score(predicted_price_lightbgm, actual_price)
        }
    with open(data_path + '/output/feature_stacking_result_scores_' + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S") + '.json', 'w') as file:
            json.dump(results, file)

def get_predicted_prices_by_feature_vectors(test_df, validation_df, train_df):
    # train_df, validation_df = add_features_to_df(train_df, validation_df)
    # test_df = add_features_to_testdf(test_df)
    X_test = test_df.drop(['Unnamed: 0', 'care_instructions','images', 'image_0', 'image_1', 'image_2', 'price'],
                     axis=1)
    y_test = test_df['price']

    y_train = train_df['price']
    X_train = train_df.drop(['Unnamed: 0','images','care_instructions','image_0','image_1','image_2','price'], axis = 1)

    y_validation = validation_df['price']
    X_validation = validation_df.drop(
        ['Unnamed: 0', 'images', 'care_instructions', 'image_0', 'image_1', 'image_2', 'price'], axis=1)

    train_data, validation_data, test_data = processing.generate_encodings_for_features(X_train, X_validation, X_test)

    np.savez_compressed(data_path + '/train_features.npz', train_data)
    np.savez_compressed(data_path + '/validation_features.npz', validation_data)
    np.savez_compressed(data_path + '/test_features.npz', test_data)

    print('--Final Data Matrix--')
    print(train_data.shape, y_test.shape)
        # print(test_data.shape, row['price'].shape)
        #do price prediction and return the price

    return train_data, validation_data, test_data, y_train, y_validation, y_test

if __name__ == "__main__":
    myparser = argparse.ArgumentParser(description='Run this file to retrieve the matching images to the query image')
    myparser.add_argument('-m', '--method', action='store', type=str, help='Default will use feature stacking')
    myparser.add_argument('-a', '--text_analysis', dest='text_analysis', action='store_true')
    myparser.add_argument('-t', '--run_test', dest='run_test', action='store_true')
    myparser.set_defaults(text_analysis=False)
    myparser.set_defaults(run_test=False)

    test_df = pd.read_csv(data_path + "/test_records.csv")
    validation_df = pd.read_csv(data_path + "/validation_records.csv")
    train_df = pd.read_csv(data_path + "/train_records.csv")

    predicted_price_xgb = []
    predicted_price_ridge = []
    predicted_price_lightbgm = []
    actual_price = []

    train_data, validation_data, test_data, y_train, y_validation, y_test = get_predicted_prices_by_feature_vectors(test_df, validation_df, train_df)
    predicted_price_xgb.append(run_on_xgb_regressor(train_data, y_train, test_data))
    predicted_price_ridge.append(run_on_ridge_regression(train_data, y_train, test_data))
    predicted_price_lightbgm.append(run_on_light_bgm_regressor(train_data, y_train, test_data))

    find_rmse_mae_r2(predicted_price_xgb, predicted_price_ridge, predicted_price_lightbgm, y_test)

