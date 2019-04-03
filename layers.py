from __future__ import print_function
import numpy as np


def sig(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

def mse(y_,y, derivative = False):
    # y__ = y_[np.arange(len(y_)),y]
    # print(len(y__[0]))
    if derivative:
        return np.subtract(y_ , y)

    return np.square(y_ - y)/2
np.random.seed(37)


class Layer:

    def __init__(self):
        pass


    def forward(self,input):
        return input


    def backward(self, input, grad_output):

        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)


class sigmoid(Layer):

    def __init__(self):
        pass
    
    def forward(self, input):
        return sig(input)

    def backward(self, input, grad_output):
        return grad_output*sig(input, derivative=True)

    
class Dense(Layer):
    def __init__(self, input_units, output_units, alpha=0.4):
        self.alpha = alpha
        self.weights = np.random.normal(size = (input_units, output_units))
        self.biases = np.ones(output_units)

    def forward(self, input):
        return np.dot(input, self.weights)+self.biases

    def backward(self, input, grad_output):
        grad_input = np.dot( grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]

        self.weights = self.weights - self.alpha*grad_weights
        self.biases = self.biases-self.alpha*grad_biases

        return grad_input

class conv2d(Layer):
    def __init__(self, input_units, output_units, alpha=0.4, filter_size = (3,3), num_filters = 1, padding = false):
        self.alpha = alpha 
        self.weights = np.random.normal(size = (num_filters,filter_size[0],filter_size[1]))
        self.biases = np.ones(num_filters)

def softmax_crossentropy_with_logits(y_,y):
    logits_for_answers = y_[np.arange(len(y_)),y]
    
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(y_),axis=-1))
    
    return xentropy

def grad_softmax_crossentropy_with_logits(y_,y):
    ones_for_answers = np.zeros_like(y_)
    ones_for_answers[np.arange(len(y_)),y] = 1
    
    softmax = np.exp(y_) / np.exp(y_).sum(axis=-1,keepdims=True)
    
    return (- ones_for_answers + softmax) / y_.shape[0]
