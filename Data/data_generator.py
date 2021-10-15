import pandas as pd
import numpy as np
from pathlib import Path
import requests
import os
from glob import iglob
from platform import platform
from PIL import Image


def read_file(file_name):
    return pd.read_csv(file_name, error_bad_lines=False)


def remove_columns(df, col_names):
    df.drop(col_names, inplace=True, axis=1)
    return df


def GetAllImagesInPath(path):
    jpg_path = os.path.join(path, '**/*.jpg')
    jpeg_path = os.path.join(path, '**/*.jpeg')
    bmp_path = os.path.join(path, '**/*.bmp')
    png_path = os.path.join(path, '**/*.png')

    image_paths = []

    image_paths.extend(iglob(jpg_path, recursive=True))
    image_paths.extend(iglob(jpeg_path, recursive=True))
    image_paths.extend(iglob(bmp_path, recursive=True))
    image_paths.extend(iglob(png_path, recursive=True))

    # windows is case insensitive so we don't need to add this
    if not platform().startswith('Windows'):
        jpg_path = os.path.join(path, '**/*.JPG')
        jpeg_path = os.path.join(path, '**/*.JPEG')
        bmp_path = os.path.join(path, '**/*.BMP')
        png_path = os.path.join(path, '**/*.PNG')

        image_paths.extend(iglob(jpg_path, recursive=True))
        image_paths.extend(iglob(jpeg_path, recursive=True))
        image_paths.extend(iglob(bmp_path, recursive=True))
        image_paths.extend(iglob(png_path, recursive=True))

    return image_paths


def remove_duplicate_images(image):
    if int(image.split('_')[-1].split('.')[0]) > 0:
        os.remove(image)
        return False
    return True


def check_if_exist(path):
    return os.path.isfile()


def removeIfnotGood(image):
    try:
        im = Image.open(image)
        return False
    except IOError:
        # filename not an image file
        os.remove(image)
        return True


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__)) + '/ecommerce_data.csv'
    df = read_file(Path(file_path))
    print(df.iloc[0])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    columns_to_remove = ['crawl_timestamp', 'product_id', 'link', 'size', 'variant_sku', 'brand', 'care_instructions',
                         'dominant_material', 'title', 'actual_color', 'dominant_color', 'product_type', 'body',
                         'product_details', 'size_fit', 'complete_the_look', 'variant_price', 'is_in_stock',
                         'ideal_for', 'inventory', 'specifications']
    df = remove_columns(df, columns_to_remove)
    columns = list(df.columns.values)
    new_df = pd.DataFrame(columns)
    print(df.iloc[0])
    # df = clean_type(df)
    i = 0
    image_paths = GetAllImagesInPath(os.path.dirname(os.path.realpath(__file__)) + "/Images/")
    list_Full = df['uniq_id'].to_list()
    partial_list = []
    for image_path in image_paths:
        # res = remove_duplicate_images(image_path)
        # if res:
        uniq = image_path.split('\\')[-1].split('_')[0]
        res = removeIfnotGood(image_path)
        if not res:
            partial_list.append(uniq)
        # str = df.loc [df['uniq_id'] == uniq]
        # print(str)
        # new_df.append(str, ignore_index=True)
    to_be_removed_list = list(set(list_Full) - set(partial_list))
    new_df = df[~df['uniq_id'].isin(to_be_removed_list)]
    print(new_df.iloc[0])
    path = os.path.dirname(os.path.realpath(__file__)) + '/updated_ecommerce_data.csv'
    new_df.to_csv(path)
