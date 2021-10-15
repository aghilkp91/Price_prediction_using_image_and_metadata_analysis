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
from xgboost import XGBRegressor
import traceback

data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data'
# images_path = "/ceph/csedu-scratch/project/akaradathodi/Images/"
images_path = os.path.abspath(data_path + '/../../') + '/Images/'
test_df = pd.read_csv(data_path + "/test_records.csv")
train_df = pd.concat([pd.read_csv(data_path + "/train_records.csv"),pd.read_csv(data_path + "/validation_records.csv")])

def create_dataframe_of_retreived_images(images):
    uniq_ids = []
    for image in images:
        uniq_ids.append(image[0].split('/')[-1].split('_')[0])
    temp_df = train_df[train_df['uniq_id'].isin(uniq_ids)]
    return temp_df

def run_on_xgb_regressor(train_data, y_train, test_data):
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(train_data, y_train)
    predictions = xgb_model.predict(test_data)
    return predictions


def find_rmse_mae_r2(predicted_price, actual_price):
    predicted_price = np.array(predicted_price)
    actual_price = np.array(actual_price)

    results = {
            'MSE': mean_squared_error(predicted_price, actual_price),
            'MAE': mean_absolute_error(predicted_price, actual_price),
            'R2': r2_score(predicted_price, actual_price)
        }
    with open(data_path + '/output/CBIR_result_scores_' + datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S") + '.json', 'w') as file:
            json.dump(results, file)

def get_predicted_prices_for_CBIR():
    #loop through each query image and predict the prices
    predicted_price = []
    actual_price = []
    X_last = test_df.drop(['Unnamed: 0', 'uniq_id', 'care_instructions','images', 'image_0', 'image_1', 'image_2', 'price'],
                      axis=1)
    for index, row in test_df.iterrows():
        query_img = images_path + row['uniq_id'] + '_0' + '.jpg'
        images = retrieveImage.retrieve_images_by_CBIR(query_img, database_path, False)
        actual_price.append(row['price'])
        if images:
            temp_df = create_dataframe_of_retreived_images(images)
            # print(temp_df.uniq_id)
            y = temp_df['price']
            X = temp_df.drop(['Unnamed: 0','uniq_id','care_instructions','image_0','image_1','image_2','price'], axis = 1)
            try:
                train_data, test_data = processing.generate_encodings(X, X_last.iloc[[index], :])
            except Exception as exp:
                print(exp)
                print(traceback.format_exc())
                exit(0)
            print('--Final Data Matrix--')
            print(train_data.shape, y.shape)
        # print(test_data.shape, row['price'].shape)
        #do price prediction and return the price
            predicted_price.append(run_on_xgb_regressor(train_data, y, test_data))
            # break
        else:
            predicted_price.append(0)
    return predicted_price, actual_price

if __name__ == "__main__":
    myparser = argparse.ArgumentParser(description='Run this file to retrieve the matching images to the query image')
    myparser.add_argument('-m', '--method', action='store', type=str, help='CBIR for content based image retrieval')
    myparser.add_argument('-a', '--text_analysis', dest='text_analysis', action='store_true')
    myparser.add_argument('-t', '--run_test', dest='run_test', action='store_true')
    myparser.set_defaults(text_analysis=False)
    myparser.set_defaults(run_test=False)

    args = vars(myparser.parse_args())
    if args['method'] == 'CBIR':
        database_path = retrieveImage.get_database_path({})
        predicted_price, actual_price = get_predicted_prices_for_CBIR()
        find_rmse_mae_r2(predicted_price, actual_price)

    # elif args['method'] == '':
    #     # do something+
