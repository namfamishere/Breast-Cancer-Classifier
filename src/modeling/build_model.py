from imutils import paths

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense

from tensorflow.keras import backend as K

class CancerNet:
    @staticmethod
    def build_model(width, height, depth, classes):
        # initialize model along with input shape to be "channels last" and channels dimension ifself
        model = Sequential()
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        # CONV => RELU => POOL
        model.add(SeparableConv2D(32, (3,3), padding='same', activation='relu', input_shape = input_shape))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        # (CONV => RELU => POOL) * 2

        model.add(SeparableConv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(SeparableConv2D(64, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # (CONV => RELU => POOL) * 3
        model.add(SeparableConv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))

        model.add(SeparableConv2D(128, (3, 3), padding="same", activation='relu'))
        model.add(BatchNormalization(axis=chan_dim))
        model.add(SeparableConv2D(128, (3, 3), padding="same", activation='relu'))

        model.add(BatchNormalization(axis=chan_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # 
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax
        model.add(Dense(classes, activation='softmax'))

        return model

if __name__ == "__main__":

    model = CancerNet.build_model(48, 48, 3, 2)
    print("Getting the model architecture...")
    model.summary()
    


