from __future__ import print_function
from keras import backend as K
import tensorflow as tf
mnist = tf.keras.datasets.mnist
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
epochs = 32
batch_size = 32

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((x_train.shape[0],28,28,1))
x_test = x_test.reshape((x_test.shape[0],28,28,1))


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters = 32, kernel_size=(7,7), padding='Same', activation = 'relu', input_shape = (28,28,1)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# summarize history for accuracy
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.4, verbose=2)
model.save_weights("mnist_1.h5")

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
