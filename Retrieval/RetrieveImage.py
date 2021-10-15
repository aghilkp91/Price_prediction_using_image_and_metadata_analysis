import os
import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
import DeepImageUtils as IU
import argparse

def get_feature_vectors(query_img_path, database_path, plotyn):
    query_img_categories = IU.PredictImageCategory(query_img_path)

# saving features and loading them when a query is need is slightly faster and less memory intensive than extracting the features dueing runtime
    feature_vectors = []
    for category in query_img_categories:
        category_path = os.path.join(database_path, category, category + '.npz')
  
        if not os.path.isfile(category_path):
            continue

    # load the saved features of that category
        loaded_feature_vectors = np.load(category_path, allow_pickle=True)
        loaded_feature_vectors = list(loaded_feature_vectors['arr_0'])

    # only add every image once (using the image name as a reference since that shouldn't be duplicated)
        for loaded_feature_vector in loaded_feature_vectors:
            if IU.Path2Name(loaded_feature_vector[0]) in (IU.Path2Name(feature_vector[0]) for feature_vector in feature_vectors):
                continue
      
            feature_vectors.append(loaded_feature_vector)

    if not feature_vectors:
        # no maching images found in the database
        if plotyn:
            print('No matching images found.')
            # Display original image and exit script
            plt.figure()
            query_img = IU.OpenImage(query_img_path)
            plt.axis('off')
            plt.title('No matching images found')
            plt.imshow(query_img)
            plt.show()
            exit()
        else:
            return False
    else:
        return feature_vectors


def getTilteforMatchingImage(i):
    i = str(i)
    if i.endswith('1'):
        return i + 'st matching image\n'
    if i.endswith('2'):
        return i + 'nd matching image'
    if i.endswith('3'):
        return i + 'rd matching image'

    return i + 'th matching image'

def sort_and_return_images(query_img_path, feature_vectors, plotyn):
    img_features_vector = IU.CreateImageFeaturesVector(query_img_path)
    # Sort the list of feature vectors based on the maximum cosine similarity (min distance) to the query image's feature vector
    feature_vectors.sort(key=lambda feature_vector: distance.cosine(feature_vector[1], img_features_vector))

    if plotyn:
        n_matching_images_to_show = 5
        n_cols = 2
        n_rows = 3


        # Display all results, alongside original image
        results_figure=plt.figure()

        query_img = IU.OpenImage(query_img_path)

        results_figure.add_subplot(n_rows, n_cols, 1)
        plt.axis('off')
        plt.title('Search image')
        plt.imshow(query_img)

        # show the selected number of closest images

        if n_matching_images_to_show > n_rows * n_cols:
            n_matching_images_to_show = n_rows * n_cols

        if n_matching_images_to_show > len(feature_vectors):
            n_matching_images_to_show = len(feature_vectors)
        else:
            feature_vectors = feature_vectors[:n_matching_images_to_show]

        for i in range(0, n_matching_images_to_show):
            results_figure.add_subplot(n_rows, n_cols, i+2)
            plt.axis('off')
            plt.title(getTilteforMatchingImage(i+1))
            # read image file
            match = IU.OpenImage(feature_vectors[i][0])
            plt.imshow(match)

        plt.show()
        exit()
    else:
        return feature_vectors

def retrieve_images_by_CBIR(query_img_path, database_path, plotyn):
    if not os.path.isdir(database_path):
        print('No such directory:', database_path)
        return False

    if not os.path.isfile(query_img_path):
        print('No such file:', query_img_path)
        return False
    vectors = get_feature_vectors(query_img_path, database_path, plotyn)
    if vectors:
        return sort_and_return_images(query_img_path, vectors, plotyn)
    else:
        print('No matching images found.')
        return False

def get_database_path(args):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    databasepath = current_folder + '/database'

    if 'database_path' in args.keys():
        database_path = os.path.realpath(args.database_path)
    elif os.path.isdir(databasepath):
        database_path = databasepath
    else:
        database_path = input('Enter the path to the root directory of the database (named databse): \n')
        database_path = os.path.realpath(database_path)

    return database_path

if __name__ == "__main__":
    myparser = argparse.ArgumentParser(description='Run this file to retrieve the matching images to the query image')
    myparser.add_argument('-d', '--database_path', action='store', type=str)
    myparser.add_argument('-q', '--query_image', action='store', type=str)
    myparser.add_argument('-n', '--no_plots', dest='plotyn',action='store_false')
    myparser.set_defaults(plotyn=True)

    args = vars(myparser.parse_args())

    database_path = get_database_path(args)

    if args['query_image']:
        query_img_path = args['query_image']
    else:
        query_img_path = input('Enter query image path:\n')
        query_img_path = os.path.realpath(query_img_path)

    res = retrieve_images_by_CBIR(query_img_path, database_path, args['plotyn'])
    if not res:
        exit(1)
