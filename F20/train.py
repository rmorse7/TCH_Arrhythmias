import time
import os

from data_generator import get_train_valid_generator
#from losses import make_loss, dice_coef_clipped, binary_crossentropy, dice_coef, ceneterline_loss
import tensorflow as tf
import time
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

import matplotlib.pyplot as plt


EPOCHS = 20
BATCH_SIZE = 16#8
DATA_DIR = 'I:/COMP549/cwt_features_images_ecg' #I:/COMP549/cwt_features_images'
LOG_DIR = "./log"
VAL_SIZE = 0.15

def summarize_diagnostics(history):
    # you could use this function to plot the result 
    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    # plot loss
    ax[0].set_title('Loss Curves', fontsize=20)
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='test')
    ax[0].set_xlabel('Epochs', fontsize=15)
    ax[0].set_ylabel('Loss', fontsize=15)
    ax[0].legend(fontsize=15)
    # plot accuracy
    ax[1].set_title('Classification Accuracy', fontsize=20)
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='test')
    ax[1].set_xlabel('Epochs', fontsize=15)
    ax[1].set_ylabel('Accuracy', fontsize=15)
    ax[1].legend(fontsize=15)

def train():
    model = twoLayerCNN(input_size=(32,120,2))
    #model = VGG(input_shape=(128,480,2))
    model.summary()
#    model.load_weights(pre_model_path)
    # model.compile(optimizer=Adam(lr=3e-4), loss=make_loss('bce_dice'),
    #               metrics=[dice_coef, binary_crossentropy, ceneterline_loss, dice_coef_clipped])
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer= Adam(lr=3e-5),
              metrics=['accuracy'])
    print("got twolayerCNN")
    model_name = 'twolayerCNN_ecg-{}'.format(int(time.time()))

    if not os.path.exists("./results/"):
        os.mkdir('./results')
    if not os.path.exists("./weights/"):
        os.mkdir('./weights')
    save_model_weights = "./weights/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5"
    print('Fitting model...')
    start_time = time.time()
    tensorboard = TensorBoard(log_dir = LOG_DIR, write_images=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3, verbose=1, mode='min')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_weights,
                            monitor="val_loss",
                            mode = "min",
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)

    csv_logger = CSVLogger('./results/{}_train.log'.format(model_name))
    train_gen, valid_gen, num_train, num_valid = get_train_valid_generator(data_dir=DATA_DIR,batch_size=BATCH_SIZE,val_size = VAL_SIZE)
    history = model.fit(x = train_gen, 
                        validation_data=valid_gen,
                        epochs=EPOCHS,
                        steps_per_epoch=(num_train+BATCH_SIZE-1)//BATCH_SIZE,
                        validation_steps=(num_valid+BATCH_SIZE-1)//BATCH_SIZE,
                        callbacks=[earlystop, checkpoint, tensorboard, csv_logger])

    end_time = time.time()
    print("Training time(h):", (end_time - start_time) / 3600)
    summarize_diagnostics(history)

if __name__ == "__main__":
    train()
