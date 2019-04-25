from __future__ import print_function
import numpy as np
import os

import keras
# from keras.applications import ResNet50
from keras.models import Model,Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,Conv2DTranspose
from keras.layers import GlobalAveragePooling2D, GaussianNoise, Input, Dropout, concatenate
from keras import optimizers
# from keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def net(shape, pretrained_weights = None):

    inp = Input(shape)

    conv1 = Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation = 'relu')(inp)
    conv1 = Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation = 'relu')(conv1)
    pool1 = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='Same')(conv1)

    conv2 = Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation = 'relu')(pool1)
    conv2 = Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation = 'relu')(conv2)
    pool2 = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='Same')(conv2)

    conv3 = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(pool2)
    conv3 = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(conv3)
    pool3 = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='Same')(conv3)

    conv4 = Conv2D(filters = 256, kernel_size=(3,3), padding='Same', activation = 'relu')(pool3)
    conv4 = Conv2D(filters = 256, kernel_size=(3,3), padding='Same', activation = 'relu')(conv4)
    pool4 = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='Same')(conv4)

    conv5 = Conv2D(filters = 512, kernel_size=(3,3), padding='Same', activation = 'relu')(pool4)
    conv5 = Conv2D(filters = 512, kernel_size=(3,3), padding='Same', activation = 'relu')(conv5)
    pool5 = MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='Same')(conv5)

    up6 = Conv2DTranspose(filters = 256, kernel_size = (3,3), strides = (2,2), padding = "same")(pool5)
    merge6 = concatenate([pool4,up6], axis = 3)
    conv6 = Conv2D(filters = 256, kernel_size=(3,3), padding='Same', activation = 'relu')(merge6)


    up7 = Conv2DTranspose(filters = 128, kernel_size = (3,3), strides = (2,2), padding = "same")(conv6)
    merge7 = concatenate([pool3,up7], axis = 3)
    conv7 = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(merge7)


    up8 = Conv2DTranspose(filters = 64, kernel_size = (3,3), strides = (2,2), padding = "same")(conv7)
    merge8 = concatenate([pool2,up8], axis = 3)
    conv8 = Conv2D(filters = 64, kernel_size=(3,3), padding='Same', activation = 'relu')(merge8)


    up9 = Conv2DTranspose(filters = 32, kernel_size = (3,3), strides = (2,2), padding = "same")(conv8)
    merge9 = concatenate([pool1,up9], axis = 3)
    conv9 = Conv2D(filters = 32, kernel_size=(3,3), padding='Same', activation = 'relu')(merge9)

    up10 = Conv2DTranspose(filters = 3, kernel_size = (3,3), strides = (2,2), padding = "same")(conv9)
    merge10 = concatenate([inp,up10], axis = 3)
    conv10 = Conv2D(filters = 1, kernel_size=(3,3), padding='Same', activation = 'sigmoid')(merge10)

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return Model(inp,conv10)
