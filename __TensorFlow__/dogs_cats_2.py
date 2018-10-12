# Choose the underlying compiler - tensorflow or theano
import json
import os 
with open(os.path.expanduser('~') + "/.keras/keras.json","r") as f:
    compiler_data = json.load(f)
compiler_data["backend"] = "tensorflow"
compiler_data["image_data_format"] = "channels_last"  
with open(os.path.expanduser('~') + '/.keras/keras.json', 'w') as outfile:
    json.dump(compiler_data, outfile)

# import all the required packages
import numpy as np
from keras.models import Model
import keras.backend as K
import keras
from keras import layers
from keras import metrics
from keras.layers import Dense, Convolution2D, BatchNormalization, Dropout, Flatten, AveragePooling2D, Activation, Input, MaxPooling2D
from keras.optimizers import SGD
if False:
    from keras.engine.topology import Merge

defined_metrics = []
defined_loss = ''



# Load data from folder images

import os
import PIL
from PIL import Image
import numpy as np
from keras.utils import np_utils

train_data = []
train_label = []
train_label_names = []
trainRootPath = '/Path_to_traindataset'
allFolders = os.listdir(trainRootPath)
for i, cfolder in enumerate(allFolders):
    allFiles = os.listdir(trainRootPath + cfolder)
    for j, cimage in enumerate(allFiles):
        img = Image.open( trainRootPath + cfolder + '/' + cimage ).convert('LA')
        img = img.resize((299,299), PIL.Image.ANTIALIAS)
        img.load()
        cdata = np.asarray( img, dtype="int32" )
        train_data.append(cdata)
        train_label.append(i)
        train_label_names.append(cfolder)

train_data = np.array(train_data)
train_label = np.array(train_label)
train_label = np_utils.to_categorical(train_label, len(set(train_label)))

val_data = []
val_label = []
val_label_names = []
if('/Path_to_valdataset'):
    valRootPath = '/Path_to_valdataset'
    allFolders = os.listdir(valRootPath)
    for i, cfolder in enumerate(allFolders):
        allFiles = os.listdir(valRootPath + cfolder)
        for j, cimage in enumerate(allFiles):
            img = Image.open( valRootPath + cfolder + '/' + cimage ).convert('LA')
            img = img.resize((299,299), PIL.Image.ANTIALIAS)
            img.load()
            cdata = np.asarray( img, dtype="int32" )
            val_data.append(cdata)
            val_label.append(i)
            val_label_names.append(cfolder)

    val_data = np.array(val_data)
    val_label = np.array(val_label)
    val_label = np_utils.to_categorical(val_label, len(set(val_label)))


test_data = []
test_label = []
test_label_names = []
if('/Path_to_testdataset'):
    testRootPath = '/Path_to_testdataset'
    allFolders = os.listdir(testRootPath)
    for i, cfolder in enumerate(allFolders):
        allFiles = os.listdir(testRootPath + cfolder)
        for j, cimage in enumerate(allFiles):
            img = Image.open( testRootPath + cfolder + '/' + cimage ).convert('LA')
            img = img.resize((299,299), PIL.Image.ANTIALIAS)
            img.load()
            cdata = np.asarray( img, dtype="int32" )
            test_data.append(cdata)
            test_label.append(i)
            test_label_names.append(cfolder)

    test_data = np.array(test_data)
    test_label = np.array(test_label)
    test_label = np_utils.to_categorical(test_label, len(set(test_label)))

print(train_data.shape)
batch_input_shape_Custom = train_data.shape[1:]
train_batch_size = 16

# Choose the hardware platform - GPU or CPU
import tensorflow as tf
if('cpu' in 'CPU'.lower()):
    device_id = '/cpu'
if('gpu' in 'CPU'.lower()):
    device_id = '/gpu:0'
if('multigpu' in 'CPU'.lower()):
    device_id = '/gpu:1'

with tf.device(device_id):

    #Input Layer
    Custom_0 = Input(shape=batch_input_shape_Custom)
    #Convolution2D Layer
    Convolution2D_1 = Convolution2D(64, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.c18db6db')(Custom_0)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_1 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.c18db6db')(Convolution2D_1)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_2 = Activation('relu', name = 'ReLU.7e9f24ad')(Convolution2D_1)
    #Dropout Layer
    Dropout_3 = Dropout(0.35, name = 'Dropout.69d8f55e')(ReLU_2)
    #Convolution2D Layer
    Convolution2D_4 = Convolution2D(64, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.3a9ebc96')(Dropout_3)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_4 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.3a9ebc96')(Convolution2D_4)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_5 = Activation('relu', name = 'ReLU.bda49317')(Convolution2D_4)
    #Pooling2D Layer
    Pooling2D_6 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.79b7880e')(ReLU_5)
    #Pooling2D Layer
    Pooling2D_7 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.e4e8b64e')(Pooling2D_6)
    #Convolution2D Layer
    Convolution2D_8 = Convolution2D(128, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.2174a78f')(Pooling2D_7)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_8 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.2174a78f')(Convolution2D_8)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_9 = Activation('relu', name = 'ReLU.ae7c0ba5')(Convolution2D_8)
    #Dropout Layer
    Dropout_10 = Dropout(0.35, name = 'Dropout.c100b76a')(ReLU_9)
    #Convolution2D Layer
    Convolution2D_11 = Convolution2D(128, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.ca77e78d')(Dropout_10)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_11 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.ca77e78d')(Convolution2D_11)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_12 = Activation('relu', name = 'ReLU.6011e9ac')(Convolution2D_11)
    #Pooling2D Layer
    Pooling2D_13 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.68bcd3de')(ReLU_12)
    #Pooling2D Layer
    Pooling2D_14 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.e659d7ac')(Pooling2D_13)
    #Convolution2D Layer
    Convolution2D_15 = Convolution2D(256, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.a4814e4')(Pooling2D_14)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_15 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.a4814e4')(Convolution2D_15)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_16 = Activation('relu', name = 'ReLU.a70b041')(Convolution2D_15)
    #Dropout Layer
    Dropout_17 = Dropout(0.35, name = 'Dropout.db88dc2c')(ReLU_16)
    #Convolution2D Layer
    Convolution2D_18 = Convolution2D(256, 3, 3, init = 'lecun_uniform', border_mode = 'valid', subsample = (1, 1), dim_ordering = 'tf', bias = False, name = 'Convolution2D.d2e72e06')(Dropout_17)
    #Batch Normalization Layer
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    Convolution2D_18 = BatchNormalization(axis=bn_axis,name='bn_Convolution2D.d2e72e06')(Convolution2D_18)
    #Rectification Linear Unit (ReLU) Activation Layer
    ReLU_19 = Activation('relu', name = 'ReLU.45fb4db9')(Convolution2D_18)
    #Pooling2D Layer
    Pooling2D_20 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.a3ec2ea8')(ReLU_19)
    #Pooling2D Layer
    Pooling2D_21 = MaxPooling2D(pool_size = (3, 3), border_mode = 'valid', strides = (3, 3), name = 'Pooling2D.969f8798')(Pooling2D_20)
    #Flatten Layer
    Flatten_22 = Flatten(name = 'Flatten.497d859e')(Pooling2D_21)
    #Dense or Fully Connected (FC) Layer
    Dense_23 = Dense(1024, init = 'lecun_uniform', bias = True, name = 'Dense.4fac24ec')(Flatten_22)
    #Sigmoid Activation Layer
    Sigmoid_24 = Activation('sigmoid', name = 'Sigmoid.5cde8524')(Dense_23)
    #Dropout Layer
    Dropout_25 = Dropout(0.5, name = 'Dropout.151333e6')(Sigmoid_24)
    #Dense or Fully Connected (FC) Layer
    Dense_26 = Dense(512, init = 'lecun_uniform', bias = True, name = 'Dense.fe94335a')(Dropout_25)
    #Sigmoid Activation Layer
    Sigmoid_27 = Activation('sigmoid', name = 'Sigmoid.ace05b6e')(Dense_26)
    #Dropout Layer
    Dropout_28 = Dropout(.3, name = 'Dropout.e7f5e01d')(Sigmoid_27)
    #Dense or Fully Connected (FC) Layer
    Dense_29 = Dense(2, init = 'uniform', bias = True, name = 'Dense.eee2cd46')(Dropout_28)
    #Softmax Activation Layer
    SoftmaxWithLoss_30 = Activation('softmax', name = 'SoftmaxWithLoss.2dad1fd6')(Dense_29)
    defined_loss = 'binary_crossentropy'

    # Define a keras model
    model = Model(input=[Custom_0, SoftmaxWithLoss_30], output = [])

    # Set the required hyperparameters    
    num_epochs = 100

    # Defining the optimizer function
    optimizer_fn = SGD(lr=0.01, momentum=0.9, decay=0.8)

    # Compile and train the model
    if not defined_metrics:
        defined_metrics=None
    if not defined_loss:
        defined_loss = 'categorical_crossentropy'
    model.compile(loss=defined_loss, optimizer=optimizer_fn, metrics=defined_metrics)
    model.fit(train_data, train_label, batch_size=train_batch_size, nb_epoch=num_epochs, verbose=1)

    # validate the model
    if (val_data):
        val_scores = model.evaluate(val_data, val_label, verbose=1)
        
    # test the model
    if (test_data):
        test_scores = model.evaluate(test_data, test_label, verbose=1)

