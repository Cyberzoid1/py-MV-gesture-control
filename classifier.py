import numpy as np
import pandas as pd
import random
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

# Disables GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class CLASSIFIER():
    def __init__(self):
        self.model_save_path = "gesture.model"
        self.model = None

    def _load_model(self):
        self.model = tf.keras.models.load_model(self.model_save_path)


    def _get_data2(self):
        data = pd.read_csv("training_data.csv",
                            names=["index","0a","0b","1a","1b","2a","2b","3a","3b","4a","4b","5a","5b","6a","6b","7a","7b","8a","8b","9a","9b","10a","10b",
                                   "11a","11b","12a","12b","13a","13b","14a","14b","15a","15b","16a","16b","17a","17b","18a","18b","19a","19b","20a","20b","label"])        
        data = data.iloc[1: , :]     # Drop first row. 
        data = data.sample(frac=1)   # Randomize
        #print(data)
        
        # Parse Data
        train_features = data.copy()
        train_features.pop('index')                   # Don't need this column
        
        # convert each column from string to float
        for col in train_features.columns:
            train_features[col] = train_features[col].astype(float)
        train_features['label'] = train_features['label'].astype(int) # Set label column to int
        train_labels = train_features.pop('label')

        print(train_features)
        print(train_labels) #.to_string(header=False))
        train_labels = pd.get_dummies(train_labels) # One-hot encode
        
        #print(train_features)
        print(train_labels)
        
        # Split dataset
        split = int(round(len(train_features)*0.8))
        print(f"Samples: {len(train_features)}  Split: {split}")
        trainX = train_features[:split]
        trainY = train_labels.iloc[:split]
        testX = train_features[split:]
        testY = train_labels.iloc[split:]

        print("trainX shape: ", trainX.shape)
        print("trainY shape: ", trainY.shape)
        
        # print("\ntrainX", trainX)
        print("trainY", trainY)
        # print("testX", testX)
        # print("testY", testY)

        return trainX, trainY, testX, testY

    def _get_model(self, no_classes=None, dropout_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(42,), name="Initial-Input"))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(50, activation='sigmoid', kernel_initializer='he_uniform', name="First-Dense"))
        #model.add(BatchNormalization())
        model.add(Dense(20, activation='sigmoid', kernel_initializer='he_uniform', name="Second-Dense"))
        model.add(Dense(no_classes, activation='softmax'))
        # compile model
        #opt = SGD(learning_rate=0.00001)
        opt = Adam(learning_rate=0.0001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary: {model.summary()}")
        return model


    def train(self): #, data):
        #trainX, trainY, testX, testY = self._get_data(data)
        trainX, trainY, testX, testY = self._get_data2()
        no_classes = trainY.shape[1]

        self.model = self._get_model(no_classes=no_classes, dropout_rate=0.2)

        # fit model
        print("\nFitting model")
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        history = self.model.fit(trainX, trainY, epochs=500, batch_size=64, validation_data=(testX, testY), callbacks=[callback], verbose=1)

        # evaluate model
        _, acc = self.model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        
        # Save model
        self.model.save(self.model_save_path)


    def __call__(self, input):
        return self.categorize(input)


    def categorize(self, input):
        #print(f"Raw input: {input}")
        if self.model is None:
            self._load_model()
        result_all = self.model.predict(input, verbose=0)
        result = tf.argmax(result_all, axis=1)
        print(f"Categorize result: {result}\tAll:{result_all}")
        return int(result[0]), result_all


# For local testing
def test():
    import pickle
    classifier = CLASSIFIER()

    # # Read from file
    with open('training_data.pickle', 'rb') as f:
        training_data = pickle.load(f)

    no_classes = 5
    classifier.train(no_classes) #, training_data)


if __name__ == "__main__":
    test()
