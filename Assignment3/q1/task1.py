from __future__ import print_function
import numpy as np
import os

batch_size = 32
epochs = 12
shape = 128,128
import keras
from keras.applications import ResNet50
from keras.models import Model,Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, GaussianNoise, Input, Dropout, concatenate
from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import keras.backend as K
from keras.backend import tf as ktf

data_dir = "../data/Q1/task1"

prev_model = ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3))

inp = model.input
x = model.output
opt2 = Dense(10, activation = 'relu', name = 'fc_2')(opt)
reg = Model(input_tensor, x)
optimizer = keras.optimizers.RMSprop()
reg.compile(loss = 'logcosh', optimizer = optimizer, metrics = ['accuracy'])

images = np.load("")
box = np.load("")

datagen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    shear_range = 0.1,
    # zoom_range = 0.1,
    height_shift_range=0.1)


datagen.fit(x_train)
