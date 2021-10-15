from shutil import copyfile
import os
import numpy as np
import DeepImageUtils as IU
import csv

def MakeDirChecked(path):
    if not os.path.isdir(path):
        os.mkdir(path)

data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data'
images_path = "/ceph/csedu-scratch/project/akaradathodi/Images/"

# root_folder = input('Enter the file where image details are mentioned: \n')
# root_folder = os.path.realpath(root_folder)

# if not os.path.isfile(image_file):
#     print('no such folder:', image_file)
#     exit(1)

# database_path = input('Enter the path to create the root folder of the database in: \n')
# database_path = os.path.join(database_path, 'database')

database_path = os.path.dirname(os.path.realpath(__file__))
database_path = os.path.join(database_path, 'database')
image_paths = []
# get the path to all image files in the root folder
paths = ["/train_records.csv", "/validation_records.csv"]
for path in paths:
    with open(data_path + path) as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        i = -1
        for row in reader:
            i += 1
            index = row[0]
            uniq_id = row[2]
            msrp = int(row[13])

            image_path = images_path + uniq_id + '_0' + '.jpg'
            if os.path.isfile(image_path):
                image_paths.append(image_path)
# image_paths = IU.GetAllImagesInPath(root_folder)

# the root create database folder
MakeDirChecked(database_path)
for img_path in image_paths:
    try:
    # predict the categories of every image in the dataset 
        categories = IU.PredictImageCategory(img_path)

    # create a folder for every category to easily separate the data reducing search times
        for category in categories:
            category_path =  os.path.join(database_path, category) + '/'
        # create the directory for the category
            MakeDirChecked(category_path)
        # copy the image to the category folder
        # could potentially just save the extracted features there and have a reference to the place of the orignal image path
        # having a reference might help with avoiding duplicates and copy times if you don't need a backup
        # and if you are scapping the web you can just link to the original image in the features database to retrun that link later
            copy_path = os.path.join(category_path, IU.Path2Name(img_path))
            copyfile(img_path, copy_path)
    except Exception as exp:
        print(exp)

# get all the created category folders in the database directory
category_folders = IU.GetAllFolderInPath(database_path)

# for every category create the feature vector for the images of that category present in the category file
total_features = []
for category_folder in category_folders:
    feature_vectors = []
    # get all image file paths' in the category's folder
    database_image_paths = IU.GetAllImagesInPath(category_folder)
    
    # extract the features of each image and append it to the feature list of that category
    # this is where you would put the link or the other path of the image instead of the local one
    for database_image_path in database_image_paths:
        uniq_id = database_image_path.split('/')[-1].split('_')[0]
        img_features = [database_image_path, IU.CreateImageFeaturesVector(database_image_path), uniq_id]
        total_features.append(img_features)
        feature_vectors.append(img_features)
  
  
    # save the features as a compressed numpy array object with the name of the category
    feature_file_name = IU.Path2Name(category_folder[:-1]) + '.npz'
    features_path = os.path.join(category_folder, feature_file_name)
    np.savez_compressed(features_path, feature_vectors)

total_features_path = data_path + '/total_features.npz'
np.savez_compressed(total_features_path, total_features)
