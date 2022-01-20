import time
import os

from data_generator import get_train_valid_generator
#from losses import make_loss, dice_coef_clipped, binary_crossentropy, dice_coef, ceneterline_loss
import tensorflow as tf
import time
import numpy as np
#import matplotlib.pyplot as plt
# -------------------------- set gpu using tf ---------------------------
# import tensorflow as tf
# import time
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# -------------------  start importing keras module ---------------------
from keras.callbacks import (ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping)
# import tensorflow.keras.backend.tensorflow_backend as K
from keras.optimizers import Adam
from CNN_models import *


###CHANGE THIS PATH TO YOUR WEIGHTS hdf5 FILE
pre_model_path = 'weights/ep018-loss0.068-val_loss0.047.hdf5'

def calculate_metrics(pre_model_path):
    """
    param:pre_model_path is the path to the model weights obtained from train.py

    Output: File prints False Positive Rate and Percentage of Arrhythmias detected
    """
    model = twoLayerCNN(input_size=(32,120,2))
    #model.summary()
    
    model.load_weights(pre_model_path)

    #model.load_weights(pre_model_path)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer= Adam(lr=3e-5),
              metrics=['accuracy', ])

    train_gen, valid_gen, num_train, num_valid = get_train_valid_generator(data_dir=DATA_DIR,batch_size=BATCH_SIZE,val_size = VAL_SIZE)

    # x_test, y_test = get_train_valid_generator(data_dir=DATA_DIR,batch_size=BATCH_SIZE,val_size = VAL_SIZE)

    count = 1

    total_right = 0
    total_wrong = 0

    false_positives = 0
    true_negatives = 0

    total_arrhythmias = 0
    arrhythmias_detected = 0

    for data_point in valid_gen:
        if count <= 16793:
            count += 1

            ##Predictions are stored as either [a, b] where a is probability of arrhythmia and b is probability of healthy
            y_prediction = model.predict(data_point[0])
            for idx in range(len(y_prediction)):
                
                if (y_prediction[idx][0] > y_prediction[idx][1] and  data_point[1][idx][0] == 1) or (y_prediction[idx][0] < y_prediction[idx][1] and data_point[1][idx][1] == 1):
                    total_right += 1
                else:
                    total_wrong += 1

                ##Classified as an Arrhythmia but actually healthy
                if y_prediction[idx][0] > y_prediction[idx][1] and data_point[1][idx][1] == 1:
                    false_positives += 1
                elif data_point[1][idx][1] == 1:
                    true_negatives += 1
                ##Classified as an arrhythmia and actually an arrhythmia
                if y_prediction[idx][0] > y_prediction[idx][1] and data_point[1][idx][1] == 0:
                    arrhythmias_detected += 1
                elif data_point[1][idx][1] == 0:
                    total_arrhythmias += 1
                
                    
        else: break



    print("Accuracy: ", total_right/(total_right+total_wrong))
    print("False Positive Rate: ", false_positives/(false_positives+true_negatives))
    print("Percent Arrhythmias accurately detected: ", arrhythmias_detected/(arrhythmias_detected + total_arrhythmias))

    return None

calculate_metrics(pre_model_path)