import math
import os
from skimage.feature import hog
import numpy as np
from PIL import Image
import  pandas as pd
from sklearn.decomposition import PCA


data_path = image_folder = os.path.dirname(os.path.realpath(__file__))
no_of_images = 0

def get_no_of_images(file_name):
    return sum(1 for line in open(data_path + '/'+ file_name))


def pca_encodeing(images, loop):
    hog_array = np.zeros((len(images), 224*224))
    print("hogarray")
    print(hog_array)

    for i, image in enumerate(images):
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                        cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
        flattened_len = int(hog_image.shape[0]) * int(hog_image.shape[1])
        pixels = np.reshape(hog_image, (-1, flattened_len))
        hog_array[i, :] = pixels
    
        if i % 1000 == 0:
            print(i)

    # pca = PCA(n_components=1)
    pca = PCA()
    pca.fit(hog_array)
    hog_images_compressed = pca.transform(hog_array)
    print(hog_images_compressed)

    # pca = PCA(n_components=200)
    pca = PCA(n_components=.9, svd_solver='full')
    pca.fit(hog_array)
    hog_images_compressed = pca.transform(hog_array)
    PCA_folder  = data_path + '\\' + 'output\\PCA\\'
    np.save(PCA_folder + "ecommerce_linreg_hog_pca_features_" + str(loop), hog_images_compressed)
    np.save(PCA_folder + "ecommerce_linreg_hog_pca_components_" + str(loop), pca.components_)

def load_images(file_name, images, im_prices):
    
    with open(data_path + '\\'+ file_name, "r") as file:
        counter = -1
        for line in file:
            if counter == -1:
                counter += 1
                continue
            line_arr = line.split(',')
            img_path = data_path + '\\' + 'Images\\' + line_arr[1]+ '_0' + '.jpg'
            curr_im = Image.open(img_path).resize((224, 224)).convert('LA')
            # img_file = imread(img_path)
            # curr_im = color.rgb2gray(img_file)
            # print(counter)
            im_prices[counter] = int(line_arr[4])
            images.append(curr_im)
            if counter % 3000 == 0: 
                # print(len(images))
                pca_encodeing(images, int(counter / 3000))
                images = []
            counter+=1
        return counter


def split_data(perc_60, perc_20):
    ecommerce_indices = np.random.permutation(no_of_images)
    ecommerce_train_indices = ecommerce_indices[:perc_60]
    ecommerce_validation_indices = ecommerce_indices[perc_60 + 1:perc_60 + perc_20]
    ecommerce_test_indices = ecommerce_indices[perc_60 + perc_20 + 1:]
    split_folder  = data_path + '/' + 'output/data_split/'
    np.save(split_folder + "ecommerce_train_indices", ecommerce_train_indices)
    np.save(split_folder + "ecommerce_validation_indices", ecommerce_validation_indices)
    np.save(split_folder + "ecommerce_test_indices", ecommerce_test_indices)

if __name__ == "__main__":
    file_name = "updated_ecommerce_data.csv"
    no_of_images = get_no_of_images(file_name)
    im_prices = np.zeros(no_of_images)
    perc_60 = math.ceil(.6 * no_of_images)
    perc_20 = math.ceil(.2 * no_of_images)
    images = []
    # load_images(file_name, images, im_prices)
    split_data(perc_60, perc_20)