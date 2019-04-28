from __future__ import print_function
import numpy as np
import os

import keras
from keras.applications import ResNet50
from keras.models import Model,Sequential
from keras.layers import Lambda, Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D, GaussianNoise, Input, Dropout, concatenate
from keras import optimizers
from keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import keras.backend as K
from keras.backend import tf as ktf


model = ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3))


inp = model.input
x = model.output
x = GlobalAveragePooling2D()(x)

reg = Dense(256, activation = 'relu')(x)
reg = Dense(16, name = 'reg')(reg)

model = Model(inp, reg)
model.summary()

model.compile(optimizer = 'adam', loss = 'logcosh', metrics = ['accuracy'])

images = np.load("images_128.npy")
points = np.load("points_128.npy")

# images = images.reshape((128,128,1))

history = model.fit(images, points, epochs=1, batch_size=32, shuffle=True,validation_split = 0.1,verbose=1)
