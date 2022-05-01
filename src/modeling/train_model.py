#import sys
#sys.path.insert(0, "./src/config")
#sys.path.insert(0, "./src/data")

#from src.modeling.build_model import *
from distutils.command.build import build
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adagrad

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, "./src/modeling")
sys.path.insert(0, "./src/config")
sys.path.insert(0, "./src/data")
sys.path.insert(0, "./src/utils")

from config import *
from build_model import *
from build_dataset import *




def train_model(model, train_gen, val_gen, train_size, val_size, batch_size, class_weight, num_epochs, callback):
    history = model.fit(x=train_gen,
                steps_per_epoch=train_size // batch_size,
                validation_data=val_gen,
                validation_steps = val_size // batch_size,
                class_weight=class_weight,
                epochs=num_epochs,
                callbacks=[callback])
    return history

def evaluate(model, test_gen):
    prediction = model.predict(x=test_gen, steps=(TEST_SIZE)//BATCH_SIZE + 1)
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

#if __name__ == "__main__":

class_weight = get_class_weight(TRAIN_PATH)
train_gen, val_gen, test_gen = get_train_val_test_generator(TRAIN_PATH, VAL_PATH, TEST_PATH)

optimizer = Adagrad(learning_rate=INIT_LEARNING_RATE, decay=INIT_LEARNING_RATE/NUM_EPOCHS)
csv_logger = CSVLogger(filename='./models/training_log.csv', append=True)

model = CancerNet.build_model(48, 48, 3, 2)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print('\nTRAINING THE MODEL...')
H = train_model(model, 
                train_gen, 
                val_gen, 
                train_size=TRAIN_SIZE, 
                val_size=VAL_SIZE, 
                batch_size=BATCH_SIZE, 
                class_weight=class_weight,
                num_epochs=NUM_EPOCHS,
                callback=csv_logger)
                
model.save('./models/model.hdf5')
print('\nEVALUATING THE MODEL...')
evaluate(model, test_gen)

print('\nGENERAIING ACCURACY AND LOSS PLOT...')

# summarize history for loss
#plt.style.use('ggplot')
#plt.figure()
plt.plot(H.history['loss'])
plt.plot(H.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.savefig('./images/loss.png')

print('Loss plot is saved in images directory.')

# summarize history for accuracy
#plt.style.use('ggplot')
#plt.figure()
plt.plot(H.history['accuracy'])
plt.plot(H.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='best')
plt.savefig('./images/accuracy.png')

print('Accuracy plot is saved in images directory.')
