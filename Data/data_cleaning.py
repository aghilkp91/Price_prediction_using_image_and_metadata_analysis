import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os

image_folder = os.path.dirname(os.path.realpath(__file__)) + "/Images/"
 
def read_file(file_name):
    return pd.read_csv(file_name, error_bad_lines=False)

def remove_columns(df, col_names):
    df.drop(col_names, inplace=True, axis=1)
    return df

def download_image(df):
    local_image_path = []
    for index, row in df.iterrows():
        images = [x.strip() for x in row['images'].split('|')]
        print(images)
        for index, image in enumerate(images):
            try:
                img_data = requests.get(image).content
                image_name = image_folder + row['uniq_id'] + '_' + str(index) + '.' + image.split('.')[-1]
                with open(image_name, 'wb') as handler:
                    handler.write(img_data)
            except Exception as exp:
                print(exp)
        #     break
        # break


def clean_type(df):
    for row in df.iterrows():
        row['type'] = row['type'].split('/')[2]
    return df

if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/ecommerce_data.csv'
    df = read_file(Path(file_path))
    print(df.iloc[0])
    columns_to_remove = ['crawl_timestamp','product_id','link','size','variant_sku','brand','care_instructions','dominant_material','title','actual_color','dominant_color','product_type','body','product_details','size_fit','complete_the_look','variant_price','is_in_stock', 'ideal_for','inventory','specifications']
    df = remove_columns(df, columns_to_remove)
    print(df.iloc[0])
    # df = clean_type(df)
    download_image(df)
    df.to_csv(os.path.dirname(os.path.realpath(__file__)) + '/updated_ecommerce_data.csv')
