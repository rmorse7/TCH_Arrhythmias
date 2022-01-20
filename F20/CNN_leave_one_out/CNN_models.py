
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
# from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
# from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
import tensorflow as tf
# from matplotlib.pyplot import imshow


import keras.backend as K


def convolutional_block(X, f, filters, stage, block, s = 1,change_dimension=False):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### SHORTCUT PATH #### (≈2 lines)
    if change_dimension:
        X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), padding = 'valid',strides = (s,s), name = conv_name_base + '1',
                            kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)




    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    
    return X

def ResNet50(input_shape=(256, 256, 3), classes=2):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = X_input#ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters=[64, 64, 256], stage=2, block='a', s=1,change_dimension=True)
    X = convolutional_block(X, f = 3, filters=[64, 64, 256], stage=2, block='b')
    X = convolutional_block(X, f = 3, filters=[64, 64, 256], stage=2, block='c')

    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2,change_dimension=True)
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage=3, block='b')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage=3, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2,change_dimension=True)
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='b')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='c')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='e')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2,change_dimension=True)
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='b')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = AveragePooling2D((2,2), name="avg_pool")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model

# model = ResNet50(input_shape=(256, 256, 3), classes=2)
# model.summary()

def vgg_block(input,n_filters,n_conv):
    for _ in range(n_conv):
        input = Conv2D(n_filters,(3,3),padding='same',activation='relu')(input)
    output = MaxPooling2D((2,2),strides=(2,2))(input)

    return output

def VGG(input_shape=(256, 256, 3), classes=2):
    X_input = Input(input_shape)

    X = vgg_block(X_input,64,3)
    X = vgg_block(X,128,3)
    X = vgg_block(X,256,3)
    X = vgg_block(X,512,3)

    X = Flatten()(X)
    X = Dense(1000,activation='relu')(X)
    X = Dense(classes, activation='softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='VGG')
    return model
# model = VGG(input_shape=(128,480,2))
# model.summary()

def twoLayerCNN(input_size, filters1=32, filters2=64, dropout=True, maxpooling=True):
    inputs = Input(shape=input_size)

    conv1 = Conv2D(filters1, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_size)(inputs)
    print("conv1 shape :", conv1.shape)
    if maxpooling:
        conv1 = MaxPooling2D((2, 2))(conv1)
    if dropout:
        conv1 = Dropout(0.5)(conv1)

    conv2 = Conv2D(filters2, kernel_size=(3,3), activation='relu', padding='same')(conv1)
    print("conv2 shape: ", conv2.shape)
    if maxpooling:
        conv2 = MaxPooling2D((2, 2))(conv2)
    if dropout:
        conv2 = Dropout(0.5)(conv2)

    flatten1 = Flatten()(conv2)

    outputs = Dense(2, activation='softmax')(flatten1)
  
    model = Model(inputs, outputs)

  #model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  
    return model

