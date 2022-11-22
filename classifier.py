import tensorflow as tf
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


class CLASSIFIER():
    def __init__(self):
        pass

    def _get_data(self):
        return trainX, trainY, testX, testY

    def _get_model(self, len_classes=5, dropout_rate=0.2):
        model = Sequential()
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform', input_shape=(None, None, 1)))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(len_classes, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        #print(f"Model Summary: {model.summary()}")
        return model

    def train(self):
        model = self._get_model()
        trainX, trainY, testX, testY = self._get_data()
        # fit model
        history = model.fit(trainX,trainY, epochs=100, batch_size=32, validation_data=(testX, testY), verbose=1)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))

    def __call__(self, input):
        return self.categorize(input)
        
    def categorize(self, input):
        result = None
        return result
