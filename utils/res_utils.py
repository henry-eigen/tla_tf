import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras.engine.topology import Layer

from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate 
from keras.layers import Dropout, Convolution2D, AveragePooling2D, Subtract
from keras.layers.core import Lambda, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers import add as combine

from keras import regularizers
from keras.regularizers import l2

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

def network_end(input_shape, num_classes=10):
    inpt = Input(shape=input_shape)
    x = AveragePooling2D(pool_size=8)(inpt)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)
    return Model(inpt, outputs)


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    temp = inputs
    temp = conv(temp)
    
    if batch_normalization:
        temp = BatchNormalization()(temp)  
    if activation is not None:
        temp = Activation(activation)(temp)
   
    return temp

 
def resnet_stack(stack, num_filters, num_res_blocks, input_shape):
    
    inpt = Input(shape=input_shape)
    strides = 1 if stack == 0 else 2
    
    if stack == 0:
        x = resnet_layer(inpt)
    
        y = resnet_layer(inputs=x,
                         num_filters=num_filters,
                         strides=strides)
        
    else:
        y = resnet_layer(inputs=inpt,
                 num_filters=num_filters,
                 strides=strides)
        
    
    y = resnet_layer(inputs=y,
                     num_filters=num_filters,
                     activation=None)
        
    if stack != 0:
        x = resnet_layer(inputs=inpt, num_filters=num_filters,
                         kernel_size=1, strides=strides,
                         activation=None, batch_normalization=False)
    
    x = combine([x, y])
    x = Activation('relu')(x)
    
    strides = 1

    for res_block in range(num_res_blocks - 1):
        y = resnet_layer(inputs=x,
                         num_filters=num_filters,
                         strides=strides)
        y = resnet_layer(inputs=y,
                         num_filters=num_filters,
                         activation=None)

        x = combine([x, y])
        x = Activation('relu')(x)

    return Model(inpt, x)

def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 90:
        lr *= 0.5e-3
    elif epoch > 80:
        lr *= 1e-3
    elif epoch > 60:
        lr *= 1e-2
    elif epoch > 40:
        lr *= 1e-1
    return lr
