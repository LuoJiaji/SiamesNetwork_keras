# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:34:05 2019

@author: Bllue
"""

import keras
import numpy as np
import matplotlib.pyplot as plt

import random

from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import load_model
from keras.optimizers import SGD 
from keras.utils import np_utils

num_classes = 8
epochs = 40

def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)
    return Model(input, x)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
input_shape = x_train.shape[1:]

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(num_classes)]
train_digit_indices = np.hstack([digit_indices[0],digit_indices[1],digit_indices[2],
                                 digit_indices[3],digit_indices[4],digit_indices[5],
                                 digit_indices[6],digit_indices[7]])
train_digit_indices = np.sort(train_digit_indices)
x_train = x_train[train_digit_indices]
y_train = y_train[train_digit_indices]


digit_indices = [np.where(y_test == i)[0] for i in range(num_classes)]
digit_indices = digit_indices[:num_classes]
#te_pairs, te_y = create_pairs(x_test, digit_indices)
test_digit_indices = np.hstack([digit_indices[0],digit_indices[1],digit_indices[2],
                                digit_indices[3],digit_indices[4],digit_indices[5],
                                digit_indices[6],digit_indices[7]])
test_digit_indices = np.sort(test_digit_indices)
x_test = x_test[test_digit_indices]
y_test = y_test[test_digit_indices]

y_train = np_utils.to_categorical(y_train, num_classes=8)
y_test = np_utils.to_categorical(y_test, num_classes=8)


input_shape = x_train.shape[1:]

base_network = create_base_network(input_shape)
input_a = Input(shape=input_shape)
output = base_network(input_a)

model = Model(input_a, output)

model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])

base_network.summary()
model.summary()
model.fit(x_train,y_train,batch_size=64,epochs=5,validation_data=(x_test,y_test))
