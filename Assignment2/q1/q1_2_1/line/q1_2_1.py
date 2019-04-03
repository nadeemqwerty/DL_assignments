from __future__ import print_function
from keras import backend as K
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import *

epochs = 16
batch_size = 64

x_train = np.load("here/train.npy")
y_train = np.load("here/Y_train.npy")
x_test = np.load("here/test.npy")
y_test = np.load("here/Y_test.npy")

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters = 32, kernel_size=(7,7), padding='Same', activation = 'relu', input_shape = (28,28,3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dense(96, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# summarize history for accuracy
model.summary()


datagen = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.1,
    shear_range = 0.1,
    zoom_range = 0.1,
    height_shift_range=0.1)


datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size, epochs=epochs, validation_data=(x_test,y_test))

model.save_weights("line_1.h5")

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig("accuracy.jpg")

plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.savefig("loss.jpg")
plt.show()


# score = model.evaluate(x_test, y_test)
pr =model.predict(x_test)
model.evaluate(x_test, y_test)


pred = []
for i in pr:
    x=np.argmax(i)
    pred.append(x)
pred = np.array(pred)

truth = []
for i in y_test:
    x=np.argmax(i)
    truth.append(x)
truth = np.array(truth)
# truth.shape

f1 = f1_score(truth, pred, average="macro")

cm = np.array(confusion_matrix(truth,pred),dtype = np.float32)

np.save("confusion_matrix.npy",cm)
plt.matshow(cm)
plt.colorbar()
plt.savefig("confusion.jpg")
