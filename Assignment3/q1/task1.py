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

batch_size = 32
epochs = 16
shape = (128,128,3)

data_dir = "data/"

images = np.load(data_dir+"images.npy")
boxes = np.load(data_dir+"boxes.npy")
labels = np.load(data_dir+"labels.npy")


model = ResNet50(weights='imagenet',include_top=False, input_shape=(128,128,3))


inp = model.input
x = model.output
x = GlobalAveragePooling2D()(x)

classification = Dense(256, activation = 'relu')(x)
classification = Dense(3, activation = 'softmax',name='classification')(classification)

reg = Dense(256, activation = 'relu')(x)
reg = Dense(4, name = 'reg')(reg)

model = Model(inp, [classification,reg])
model.summary()

loss = ['categorical_crossentropy','logcosh']
loss_weights = [0.1, 2.0]

model.compile(optimizer = 'adam', loss = loss,loss_weights = loss_weights, metrics = ['accuracy'])

history = model.fit(images, [labels,boxes], epochs=epochs, batch_size=batch_size, shuffle=True,validation_split = 0.1,verbose=1)
