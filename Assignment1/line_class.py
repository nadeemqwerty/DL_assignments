from __future__ import print_function
import numpy as np 
np.random.seed(37)
from neauron import *
import cv2
import matplotlib.pyplot as plt
import os

def load_data(folder):
    images = []
    clss = []
    for filename in os.listdir(folder):
        if(filename.endswith(".jpg")):
            img = cv2.imread(os.path.join(folder,filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.array(img)
            img = img.ravel()
            images.append(img)
            cl =1
            filename = filename[:-4]
            for w in filename.split("_"):
                cl = cl*(int(w)+1)
            cl = cl - 1
            clss.append(cl)
        
    return images,clss

def split_data():
    images, clss = load_data("Line")
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    
    for i in range(96):
        x_train = x_train + images[i*1000:i*1000+800]
        y_train = y_train + clss[i*1000:i*1000+800]
        x_val = x_val + images[i*1000+801:i*1000+1000]
        y_val = y_val + images[i*1000+801:i*1000+1000]

    x_train = np.array(x_train,dtype = float32)/255.
    x_val = np.array(x_val,dtype = float32)/255.

    return x_train, y_train, x_val, y_val


epochs = 32

X_train, y_train, X_val, y_val = split_data()


model = []
model.append(Dense(X_train.shape[1],64,0.4))
model.append(sigmoid())
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
    
    train_log.append(np.mean(predict(model,X_train)==y_train))
    val_log.append(np.mean(predict(model,X_val)==y_val))
    
    
    print("Epoch",epoch)
    print("Train accuracy:",train_log[-1])
    print("Val accuracy:",val_log[-1])
plt.plot(train_log,label='train accuracy')
plt.plot(val_log,label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()
