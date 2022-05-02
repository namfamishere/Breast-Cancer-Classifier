from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys,os

sys.path.insert(0, "/content/drive/MyDrive/breast-cancer-classifier/src/modeling")
sys.path.insert(0, "/content/drive/MyDrive/breast-cancer-classifier/src/config")
sys.path.insert(0, "/content/drive/MyDrive/breast-cancer-classifier/src/data")
#sys.path.insert(0, "./src/utils")

from config import *
from build_model import *
from build_dataset import get_train_val_test_size, get_train_val_test_generator, get_class_weight




def evaluate(model, test_gen, test_size, batch_size):
    prediction = model.predict(x=test_gen, steps=(test_size)//batch_size + 1)
    prediction = np.argmax(prediction, axis=1)
    print(classification_report(test_gen.classes, prediction,
                                target_names=test_gen.class_indices.keys()))

    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(test_gen.classes, prediction)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print("confusion matrix", cm)
    print("accuaracy: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity\n: {:.4f}".format(specificity))


def display_training_curves(training, validation, title, subplot):
  ax = plt.subplot(subplot)
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])

def plot_loss(history):
    # summarize history for loss
    plt.plot(H.history['loss'])
    plt.plot(H.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig('/content/drive/MyDrive/breast-cancer-classifier/images/loss.png')
    print('Loss plot is saved at images directory.\n')

def plot_accuracy(hisotry):
    # summarize history for accuracy
    plt.plot(H.history['accuracy'])
    plt.plot(H.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig('/content/drive/MyDrive/breast-cancer-classifier/images/accuracy.png')
    print('Accuracy plot is saved at images directory.\n')

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') is not None and logs.get('accuracy')>=0.95
                    and logs.get('val_accuracy') is not None and logs.get('val_accuracy')>=0.9):
            print("\nReached 95% accuracy and 90% val accuracy so cancelling training!\n")
            self.model.stop_training = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters to train the model")
    parser.add_argument("-d", "--data_dir", type=Path, help="the path to splitted data directory")
    parser.add_argument("-n", "--num_epochs", type=int, default=40, help="the number of epochs")
    parser.add_argument("-b", "--batch_size", type=int,  default=32, help="batch size")
    parser.add_argument("-l", "--learning_rate", type=float, default=0.02, help="learning rate")
    args = vars(parser.parse_args())

    num_epochs = args['num_epochs']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']


    data_dir = str(args['data_dir'])
    TRAIN_PATH = os.path.sep.join([data_dir, "training"])
    VAL_PATH = os.path.sep.join([data_dir, 'validation'])
    TEST_PATH = os.path.sep.join([data_dir, "test"])



    train_size, val_size, test_size = get_train_val_test_size(TRAIN_PATH, VAL_PATH, TEST_PATH)

    class_weight = get_class_weight(TRAIN_PATH)


    train_gen, val_gen, test_gen = get_train_val_test_generator(TRAIN_PATH, VAL_PATH, TEST_PATH)

    optimizer = Adagrad(learning_rate=learning_rate, decay=learning_rate/num_epochs)
    csv_logger = CSVLogger(filename='/content/drive/MyDrive/breast-cancer-classifier/models/training_log.csv')


    callbacks = myCallback()

    model = CancerNet.build_model(48, 48, 3, 2)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


    plot_model(model, 
            to_file="/content/drive/MyDrive/breast-cancer-classifier/models/model.png", 
            show_shapes=True,
            show_layer_names=True,
            show_layer_activations=True)


    print('\nTRAINING THE MODEL...')

    H = model.fit(x=train_gen,
            steps_per_epoch=train_size // batch_size,
            validation_data=val_gen,
            validation_steps = val_size // batch_size,
            class_weight=class_weight,
            epochs=num_epochs,
            callbacks=[callbacks, csv_logger])

    
    plt.subplots(figsize=(10,10))
    plt.tight_layout()
    display_training_curves(H.history['accuracy'], H.history['val_accuracy'], 'accuracy', 211)
    display_training_curves(H.history['loss'], H.history['val_loss'], 'loss', 212)
    plt.savefig('/content/drive/MyDrive/breast-cancer-classifier/images/loss-accuracy.png')

    #plot_accuracy(H)
    #plot_loss(H)

                    
    model.save('/content/drive/MyDrive/breast-cancer-classifier/models/model.hdf5')
    print("Model is saved at models directory")

    print('\nEVALUATING THE MODEL...')
    evaluate(model, test_gen, test_size, batch_size)
 




