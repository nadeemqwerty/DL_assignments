from __future__ import print_function

import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, BatchNormalization, Input, Dropout, concatenate
from tensorflow.keras.models import Model
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import *

epochs = 16
batch_size = 64
input_shape = (28,28,1)

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))


def inception_block(x, filters):
#     last = x

    net1 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)

    net2 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)
    net2 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net2)

    net3 = Conv2D(filters = filters, kernel_size=(1,1), padding='Same', activation = 'relu')(x)
    net3 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net3)
    net3 = Conv2D(filters = filters, kernel_size=(3,3), padding='Same', activation = 'relu')(net3)

    output = concatenate([net1, net2, net3], axis=3)
    return output

input_layer = Input(shape=input_shape)
x = Conv2D(filters = 32, kernel_size=(5,5), padding='Same', activation = 'relu')(input_layer)
x = Conv2D(filters = 32, kernel_size=(5,5), padding='Same', activation = 'relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(5, 5), strides=(2,2))(x)

# last = x

x = inception_block(x,32)
x = MaxPool2D(pool_size=(5, 5), strides=(2,2),padding='Same')(x)

# x = MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)
# x = inception_block(x,64)
# x = MaxPool2D(pool_size=(3, 3), strides=(1,1),padding='Same')(x)

x = inception_block(x,64)
x = MaxPool2D(pool_size=(3, 3), strides=(1,1),padding='Same')(x)

# x = concatenate([x, last], axis=3)
# x = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = MaxPool2D(pool_size=(3, 3), strides=(1,1))(x)

# x = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = Conv2D(filters = 128, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = MaxPool2D(pool_size=(3, 3), strides=(2,2),padding='Same')(x)

# x = Conv2D(filters = 256, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = Conv2D(filters = 256, kernel_size=(3,3), padding='Same', activation = 'relu')(x)
# x = MaxPool2D(pool_size=(3, 3), strides=(1,1),padding='Same')(x)

x = Flatten()(x)

# x = Dense(2048, activation=tf.nn.relu)(x)
# x = Dropout(0.5)(x)

x = Dense(1024, activation=tf.nn.relu)(x)
x = Dropout(0.5)(x)

x = Dense(128, activation=tf.nn.relu)(x)
x = Dropout(0.5)(x)

x = Dense(10, activation=tf.nn.softmax)(x)

model = Model(input_layer, x)
model.summary()

# optimizer = adagrad(lr=0.01)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    shear_range = 3,
    zoom_range = 0.2,
    height_shift_range=0.2)


datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs, validation_data=(x_test,y_test))
model.save_weights("mnist_temp2.h5")


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig("accuracy.jpg")

plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig("loss.jpg")
plt.show()


pr =model.predict(x_test)

pred = []
for i in pr:
    x=np.argmax(i)
    pred.append(x)
pred = np.array(pred)

f1 = f1_score(y_test, pred, average="macro")

cm = np.array(confusion_matrix(y_test,pred),dtype = np.float32)


np.save("confusion_matrix.npy",cm)
plt.matshow(cm)
plt.colorbar()
plt.savefig("confusion.jpg")
