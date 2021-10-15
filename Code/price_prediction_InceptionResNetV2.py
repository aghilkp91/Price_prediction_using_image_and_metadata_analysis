import csv
import math
import os
import json

import matplotlib.pyplot as plt

import numpy as np
from datetime import *
from keras import applications
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input
from keras.initializers import glorot_uniform
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from ImageGenerator import ImageDataGenerator

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

train_prices = []
train_paths = []
validation_prices = []
validation_paths = []
test_prices = []
test_paths= []
data_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data'

images_path = "/ceph/csedu-scratch/project/akaradathodi/Images/"

num_settings = 1

hp_dropout = [0.2] * num_settings

# RMSprop
hp_lr = [0.001] * num_settings
hp_rho = [0.9] * num_settings
hp_epsilon = [1e-07] * num_settings
hp_decay = [0.0] * num_settings

# Number of hidden units
hp_hidden = [256] * num_settings

# Minibatch size
hp_mbsize = [64] * num_settings

num_epochs = 20

checkpoint_path = data_path + '/output/InceptionResNetV2_ecommerce-cnn-best.hdf5'

def setup_test_train_indices():
    with open(data_path + "/train_records.csv") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        i = -1
        for row in reader:
            i += 1
            index = row[0]
            uniq_id = row[1]
            msrp = int(row[9])

            image_path = images_path + uniq_id + '_0' + '.jpg'
            if os.path.isfile(image_path):
                train_paths.append(image_path)
                train_prices.append(int(msrp))

    with open(data_path + "/validation_records.csv") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        i = -1
        for row in reader:
            i += 1
            index = row[0]
            uniq_id = row[1]
            msrp = int(row[9])

            image_path = images_path + uniq_id + '_0' + '.jpg'
            if os.path.isfile(image_path):
                validation_paths.append(image_path)
                validation_prices.append(int(msrp))

    with open(data_path + "/test_records.csv") as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        i = -1
        for row in reader:
            i += 1
            index = row[0]
            uniq_id = row[1]
            msrp = int(row[9])

            image_path = images_path + uniq_id + '_0' + '.jpg'
            if os.path.isfile(image_path):
                test_paths.append(image_path)
                test_prices.append(int(msrp))

    train_indices = np.load(data_path + "/output/data_split/ecommerce_train_indices.npy")
    validation_indices = np.load(data_path + "/output/data_split/ecommerce_validation_indices.npy")
    test_indices = np.load(data_path + "/output/data_split/ecommerce_test_indices.npy")
    print("train_indices {}".format(len(train_indices)))
    print("test_indices {}".format(len(test_indices)))
    return train_indices,validation_indices,  test_indices


def image_generator(indices, batch_size, images, prices):
    print("Total indices : {}".format(len(indices)))
    num_batches = math.floor(int(len(images) / batch_size))
    # print(num_batches)
    while True:
        # indices = np.random.shuffle(indices)
        for batch_i in range(num_batches):
            print(batch_i)
            if batch_i == num_batches - 1:
                # special case: return as many as  possible
                start_i = batch_i * batch_size
                end_i = int(len(images)) - 1
                batch_indices = indices[start_i:end_i]

                X = np.zeros((len(batch_indices), 224, 224, 3))
                Y = np.zeros((len(batch_indices), 1))
                # print('Special Case')

            else:
                start_i = batch_i * batch_size
                end_i = start_i + batch_size
                print('start_i {} end_i {}'.format(start_i, end_i))
                batch_indices = indices[start_i:end_i]

                X = np.zeros((batch_size, 224, 224, 3))
                Y = np.zeros((batch_size, 1))
                # print('Normal Case')
            i = 0
            for index in batch_indices:
                if index > len(images) - 1:
                    continue
                print('Index : {}'.format(index))
                img = image.load_img(images[index], target_size=(224, 224))
                X[i, :, :, :] = image.img_to_array(img)
                Y[i] = prices[index]
                i += 1

            # use keras preprocessing
            # X = preprocess_input(X)

            yield (X, Y)


def run_model(train_indices, validation_indices, test_indices):
    # store the results of each setting
    train_losses = np.zeros(num_settings)
    dev_losses = np.zeros(num_settings)

    for setting in range(num_settings):
        # build the VGG16 network
        input_tensor = Input(shape=(224, 224, 3))
        # model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
        model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)


        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=(model.output_shape[1:])))

        # Output layer
        # We do random weight intialization
        top_model.add(Dropout(hp_dropout[setting]))
        top_model.add(Dense(hp_hidden[setting], activation='relu', kernel_initializer='glorot_uniform'))
        top_model.add(Dense(1, activation='linear', name='output', kernel_initializer='glorot_uniform'))

        # add the model on top of the convolutional base
        new_model = Model(inputs=model.input, outputs=top_model(model.output))

        # set the first 19 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in new_model.layers[:19]:
            layer.trainable = False

        # RMSprop optimizer
        new_model.compile(loss='mean_squared_error',
                          optimizer=RMSprop(
                              learning_rate=hp_lr[setting],
                              rho=hp_rho[setting],
                              epsilon=hp_epsilon[setting],
                              decay=hp_decay[setting]),
                          metrics=['mean_squared_error'])

        # keep a checkpoint
        checkpoint = ModelCheckpoint(checkpoint_path,
                                     monitor='val_loss',
                                     save_best_only=True,
                                     mode='min')

        minibatch_size = hp_mbsize[setting]
        print('minibatch_size : {}'.format(minibatch_size))

        train_steps = math.floor(len(train_indices) / minibatch_size)
        validation_steps = math.floor(len(validation_indices) / minibatch_size)

        # Initializing train and validation generator
        params = {
            'dim': (299, 299, 3),
            'batch_size': minibatch_size,
            'shuffle': True,
            'algo': 'InceptionResNetV2'
        }

        training_generator = ImageDataGenerator(train_paths, train_prices, **params)
        validation_generator = ImageDataGenerator(validation_paths, validation_prices, **params)

        # fine-tune the model
        history = new_model.fit(
            training_generator,
            steps_per_epoch=train_steps,
            epochs=num_epochs,
            validation_data = validation_generator,
            callbacks=[checkpoint])

        # store the training and dev losses for the last epoch (current model)
        train_losses[setting] = history.history['loss'][-1]
        dev_losses[setting] = history.history['val_loss'][-1]

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        # plt.show()
        plt.savefig(os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..') + '/Data/output/result_InceptionResNetV2_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.png')
        print("==========")

def predict_model_on_test():
        model = load_model(checkpoint_path)
        print("Running MSE, MAE and R2 to compare results")
        predicted_price = []
        actual_price = []
        minibatch_size = hp_mbsize[num_settings - 1]
        test_steps = math.ceil(len(test_paths) / minibatch_size)
        params = {
            'dim': (299, 299, 3),
            'batch_size': minibatch_size,
            'shuffle': True,
            'algo': 'InceptionResNetV2'
        }
        test_generator = ImageDataGenerator(test_paths, test_prices, **params)
        for step in range(test_steps):
            X,y = test_generator.__getitem__(step)
            curr_pred = model.predict(X)
            for entry in curr_pred:
                predicted_price.append(entry)
            for entry in y:
                actual_price.append(entry)

        predicted_price = np.array(predicted_price)
        actual_price = np.array(actual_price)

        results = {
            'MSE': mean_squared_error(predicted_price,actual_price),
            'MAE': mean_absolute_error(predicted_price, actual_price),
            'R2': r2_score(predicted_price, actual_price)
        }
        with open(data_path + '/output/InceptionResNetV2_result_scores_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.json', 'w') as file:
            json.dump(results, file)



if __name__ == "__main__":
    train_indices, validation_indices, test_indices = setup_test_train_indices()
    run_model(train_indices, validation_indices, test_indices)
    predict_model_on_test()
