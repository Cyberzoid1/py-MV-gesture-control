import tensorflow as tf
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout



class CLASSIFIER():
    def __init__(self):
        pass


    def _get_data(self, data):
        # https://stackoverflow.com/a/68620356
        # Extract features and lavels from data
        datasetX = [x[0] for x in data]
        datasetY = [x[1] for x in data]
        print(f"\nLen x: {len(datasetX)}, Len y: {len(datasetY)}")
        
        import random
        print("\nprinting labels")
        print(datasetY)
        print(random.shuffle(datasetY), "\n")
        
        # Create tensorflow dataset & shuffle
        dataset = tf.data.Dataset.from_tensor_slices((datasetX, datasetY))
        dataset = dataset.shuffle(buffer_size = 1000, seed=123, reshuffle_each_iteration=False)
        
        # Split dataset
        train_split = int(round(len(dataset)*0.8))
        print(f"Samples: {len(dataset)}  Split: {train_split}")
        train_ds = dataset.take(train_split)
        test_ds = dataset.skip(train_split)

        return train_ds, test_ds


    def _get_model(self, no_classes=5, dropout_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(42,)))
        model.add(Dense(42, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(no_classes, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary: {model.summary()}")
        return model


    def train(self, classes, data):
        model = self._get_model()
        train_ds, test_ds = self._get_data(data)
        
        print(f"train info. type: {type(train_ds)}")
        print(f"test info. type: {type(test_ds)}")
        
        print("len train_ds", len(train_ds))
        print("len test_ds", len(test_ds))

        print(train_ds)
        print(test_ds)
        
        # dataset_to_numpy = list(train_ds.as_numpy_iterator())
        # shape = tf.shape(dataset_to_numpy)
        # print(shape)
        
        # dataset_to_numpy = list(test_ds.as_numpy_iterator())
        # shape = tf.shape(dataset_to_numpy)
        # print(shape)

        # fit model
        history = model.fit(train_ds, epochs=100, batch_size=7, validation_data=test_ds, verbose=1)

        # evaluate model
        _, acc = model.evaluate(test_ds, verbose=0)
        print('> %.3f' % (acc * 100.0))


    def __call__(self, input):
        return self.categorize(input)


    def categorize(self, input):
        result = None
        return result
