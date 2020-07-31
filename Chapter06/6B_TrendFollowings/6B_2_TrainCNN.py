#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:36:56 2018

@author: jeff
RestNet codes are obatined from: https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
Modification are done to simplify
"""
'''*************************************
#1. Import libraries and key varable values
'''

import keras
from keras.layers import Dense, Conv2D,Conv1D, BatchNormalization, Activation
from keras.layers import AveragePooling2D,AveragePooling1D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
#from keras.datasets import cifar10
import numpy as np
import os
import math

import h5py
import time



folder_path = os.path.dirname(__file__)
tkr_list = ['DWX','TIPX','FLRN','CBND','SJNK','SRLN','CJNK','DWFI','EMTL','STOT','TOTL','DIA','SMEZ','XITK','GLDM','GLD','XKFS','XKII','XKST','GLDW','SYE','SYG','SYV','LOWC','ZCAN','XINA','EFAX','QEFA','EEMX','QEMM','ZDEU','ZHOK','ZJPN','ZGBR','QUS','QWLD','OOO','LGLV','ONEV','ONEO','ONEY','SPSM','SMLV','MMTM','VLU','SPY','SPYX','SPYD','SPYB','WDIV','XWEB','MDY','NANR','XTH','SHE','GAL','INKM','RLY','ULST','BIL','CWB','EBND','JNK','ITE','IBND','BWX','SPTL','MBG','BWZ','IPE','WIP','RWO','RWX','RWR','FEZ','DGT','XNTK','CWI','ACIM','TFI','SHM','HYMB','SPAB','SPDW','SPEM','SPIB','SPLG','SPLB','SPMD','SPSB','SPTS','SPTM','MDYG','MDYV','SPYG','SPYV','SLY','SLYG','SLYV','KBE','KCE','GII','KIE','KRE','XAR','XBI','GXC','SDY','GMF','EDIV','EWX','GNR','XHE','XHS','XHB','GWX','XME','XES','XOP','XPH','XRT','XSD','XSW','XTL','XTN','FEU','PSK']
#tkr_list = ['TIPX','HYMB','TFI','ULST','MBG','FLRN','SHM','STOT','SPTS','BIL','SPSB']
#tkr_list = ['TIPX']
# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
#run too long
#epochs = 200
#epoch = 2 for demo purpose
epochs = 2
data_augmentation = False
num_classes = 100

n = 3
# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

'''*************************************
#2. Define functions
'''
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]
    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

##############################################################################
'''*************************************
#3. Execute the model training
'''
# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2
    
# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# create list of batches to shuffle the data
batch_size = 32
tkr_cnt = 0
load_model_flag =True

#check if the prev step is completed before starting
while True:
    if os.path.isfile('6b1_completed.txt'):
        break
    else:
        print('sleep for 5min')
        time.sleep(60*5)

#decide if we should load a model or not
model_path = 'model'+str(version)+'.h5'
if load_model_flag and os.path.isfile(model_path):
    model = keras.models.load_model(model_path)
else:
    load_model_flag = False

#loop through the tickers
for tkr in tkr_list:
    if os.path.isfile(tkr+'6b2_completed.txt') and load_model_flag==True:
        continue
    
    #load dataset saved in the previous prepataion step    
    data_X_dir = os.path.join(folder_path,'dataset',tkr+'X_img.h5')
    data_Y_dir = os.path.join(folder_path,'dataset',tkr+'Y_label.h5')

    #start if both file exists:
    if os.path.isfile(data_X_dir) and os.path.isfile(data_Y_dir):

        h5f = h5py.File(data_X_dir,'r')
        h5f_X = h5f['X_img_ds'][:]
        h5f.close()

        h5f = h5py.File(data_Y_dir,'r')
        h5f_Y = h5f['Y_label_ds'][:]
        h5f.close()

        data_num = len(h5f_X)
        
        #calculate number of batches
        batches_list = list(range(int(math.ceil(float(data_num) / batch_size))))

        #do it at the first one
        if tkr_cnt == 0 and load_model_flag==False:
        # Input image dimensions.
            input_shape = h5f_X.shape[1:]

            if version == 2:
                model = resnet_v2(input_shape=input_shape, depth=depth,num_classes=num_classes)
            else:
                model = resnet_v1(input_shape=input_shape, depth=depth,num_classes=num_classes)

            model.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=lr_schedule(0)),
                          metrics=['accuracy'])
            model.summary()
            print(model_type)

        # Prepare model model saving directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        # Prepare callbacks for model saving and for learning rate adjustment.
        tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
        
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)

        callbacks = [tbCallBack, checkpoint, lr_reducer, lr_scheduler]
        

        # loop over batches
        for n, i in enumerate(batches_list):
            i_s = i * batch_size  # index of the first image in this batch
            i_e = min([(i + 1) * batch_size, data_num])  # index of the last image in this batch
            # read batch images and remove training mean
            images = h5f_X[i_s:i_e, ...]
            labels = h5f_Y[i_s:i_e, ...]

            # Run training, without data augmentation.
            print('Not using data augmentation.')
            model.fit(images, labels,
                      batch_size=batch_size,
                      epochs=epochs,
                      shuffle=True,
                      callbacks=callbacks)
        tkr_cnt+=1
        scores = model.evaluate(images, labels, verbose=1)
        #when model training finished for the ticket, create a file to indicate its completion
        f=open(tkr+'6b2_completed.txt','w+')
        f.write('Test loss:' +str(scores[0])+'\n')
        f.write('Test accuracy:'+str(scores[1])+'\n')
        f.close()
   
# Score trained model.
scores = model.evaluate(images, labels, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
model.save('model'+str(version)+'.h5')
