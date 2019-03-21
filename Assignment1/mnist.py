from __future__ import print_function
import numpy as np 
np.random.seed(37)
from neauron import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import cv2
epochs = 32
import keras
import matplotlib.pyplot as plt


def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)


model = []
model.append(Dense(X_train.shape[1],64,0.2))
model.append(sigmoid())
# model.append(Dense(128,64,0.4))
# model.append(sigmoid())
model.append(Dense(64,10,0.2))

def forward(model, X):
    
    activations = []
    input = X

    
    for l in model:
        activations.append(l.forward(input))
        
        input = activations[-1]
    return activations

def predict(model,X):
    logits = forward(model,X)[-1]
    return logits.argmax(axis=-1)

def train(model,X,y):
    
    layer_activations = forward(model,X)
    layer_inputs = [X]+layer_activations
    logits = layer_activations[-1]
    
    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)
    

    for layer_index in range(len(model))[::-1]:
        layer = model[layer_index]
        
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad)
        
    return np.mean(loss)

from tqdm import trange
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


train_log = []
val_log = []

for epoch in range(epochs):

    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32,shuffle=True):
        train(model,x_batch,y_batch)
    tp = predict(model,X_train)
    vp = predict(model,X_val)
    train_log.append(np.mean(tp==y_train))
    val_log.append(np.mean(vp==y_val))
    
    
    
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])
plt.plot(train_log,label='train accuracy')
plt.plot(val_log,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.savefig("accuracy.jpg")
pr =predict(model,X_test)
np.array(f1_score(y_test, pr, average="macro"),dtype = np.float32)
cm = np.array(confusion_matrix(y_test,pr),dtype = np.float32)
np.save("confusion_matrix.npy",cm)
a =cm.max()
cm = (cm/a)*255
cv2.imwrite("confusion_matrix.jpg", cm)
