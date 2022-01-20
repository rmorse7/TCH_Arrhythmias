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




### this train.py is used to train the neural network
## model parameters: set up
EPOCHS = 20
BATCH_SIZE = 16
DATA_DIR = 'resized_cwt_features_images/22/val'
TEST_DIR = 'resized_cwt_features_images/22/test'
LOG_DIR = "./log/22"
VAL_SIZE = 0.15



def train():
    '''
     train the neural network model based on cwt features
    '''
    #load model from CNN_models.py
    model = twoLayerCNN(input_size=(32,120,2))
    model.summary()

#    model.load_weights(pre_model_path)
    # model.compile(optimizer=Adam(lr=3e-4), loss=make_loss('bce_dice'),
    #               metrics=[dice_coef, binary_crossentropy, ceneterline_loss, dice_coef_clipped])

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer= Adam(lr=3e-5),
              metrics=['accuracy'])
    print("got vgg")
    model_name = '2DCNN_ecg_ppg-{}'.format(int(time.time()))

    if not os.path.exists("./results/22"):
        os.makedirs('./results/22')
    if not os.path.exists("./weights/22"):
        os.makedirs('./weights/22')
    save_model_weights = "./weights/22/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.hdf5"
    start_time = time.time()
    tensorboard = TensorBoard(log_dir = LOG_DIR, write_images=True)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5, verbose=1, mode='auto')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(save_model_weights,
                            monitor="val_accuracy",
                            mode = "max",
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)

    csv_logger = CSVLogger('./results/22/{}_train.log'.format(model_name))
    train_gen, valid_gen, num_train, num_valid = get_train_valid_generator(val_dir=DATA_DIR,test_dir=TEST_DIR,batch_size=BATCH_SIZE,val_size = VAL_SIZE)

    history = model.fit(x = train_gen, 
                        validation_data=valid_gen,
                        epochs=EPOCHS,
                        steps_per_epoch=(num_train+BATCH_SIZE-1)//BATCH_SIZE,
                        validation_steps=(num_valid+BATCH_SIZE-1)//BATCH_SIZE,
                        callbacks=[earlystop, checkpoint, tensorboard, csv_logger])

    end_time = time.time()
    print("Training time(h):", (end_time - start_time) / 3600)

if __name__ == "__main__":
    train()
