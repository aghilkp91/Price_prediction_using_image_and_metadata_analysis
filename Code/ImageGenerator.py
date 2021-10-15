import numpy as np
import keras
from tensorflow.keras import utils as np_utils
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.inception_resnet_v2 import preprocess_input as resnet_preprocess_input

class ImageDataGenerator(np_utils.Sequence):
    # Generates data for Keras
    def __init__(self, images, prices, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=10, shuffle=True, algo='VGG16'):
        # Initialization
        self.dim = dim
        self.batch_size = batch_size
        self.prices = prices
        self.images = images
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.algo = algo

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [k for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Generates data containing batch_size samples  # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        # y = np.empty((self.batch_size), dtype=int)
        y = np.zeros((self.batch_size, 1))
        try:
        # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                img = image.load_img(self.images[ID], target_size=(self.dim[0], self.dim[1]))
                X[i, :, :, :] = image.img_to_array(img)
                y[i] = self.prices[ID]
            # X[i,] = np.load('data/' + ID + '.npy')
            # Store class
            # y[i] = self.labels[ID]

        # return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
            if self.algo == 'VGG16':
                X = vgg16_preprocess_input(X)
            elif self.algo == 'InceptionResNetV2':
                # X = np.expand_dims(img, axis=0)
                X = resnet_preprocess_input(X)
        except Exception as exp:
                print(exp)
                raise
        return X, y
