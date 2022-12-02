import numpy
import random
import tensorflow as tf
#from tensorflow.keras.utils import to_categorical
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
        # Extract features and labels from data
        
        # Testing data override
        # data = [([1,2,3,4,5], 1),
        #         ([1,2,3,4,5], 1),
        #         ([1,2,2,2,5], 2),
        #         ([1,2,2,2,5], 2),
        #         ([1,3,3,3,5], 3)]

        random.shuffle(data)

        datasetX = numpy.array([x[0] for x in data])
        datasetY = numpy.array([x[1] for x in data])

        print(f"\nLen x: {len(datasetX)}, Len y: {len(datasetY)}")

        # Split dataset
        train_split = int(round(len(datasetX)*0.8))
        print(f"Samples: {len(datasetX)}  Split: {train_split}")
        trainX = datasetX[:train_split]
        trainY = datasetY[:train_split]
        testX = datasetX[train_split:]
        testY = datasetY[train_split:]
        
        # print("\nprinting datasetX elements")
        # for element in datasetX:
        #     print('t')
        #     print(element)
        
        # print("\nprinting datasetY elements")
        # for element in datasetY:
        #     print('t')
        #     print(element)

        print(type(trainX))
        print(type(trainY))
        print("trainX shape: ", trainX.shape)
        print("trainY shape: ", trainY.shape)
        return trainX, trainY, testX, testY


    def _get_model(self, no_classes=4, dropout_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(42,), name="Initial-Input"))
        model.add(Dropout(dropout_rate))
        #model.add(BatchNormalization())
        model.add(Dense(20, activation='relu', kernel_initializer='he_uniform', name="First-Dense"))
        #model.add(BatchNormalization())
        model.add(Dense(10, activation='relu', kernel_initializer='he_uniform', name="Second-Dense"))
        model.add(Dense(no_classes, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.001)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary: {model.summary()}")
        return model


    def train(self, no_classes, data):
        model = self._get_model(no_classes=no_classes, dropout_rate=0.2)
        trainX, trainY, testX, testY = self._get_data(data)

        # print("\nlen trainX", len(trainX))
        # print("len testX", len(testX))

        # print("\ntrainX", trainX)
        # print("trainY", trainY)
        # print("testX", testX)

        # fit model
        print("\nFitting model")
        history = model.fit(trainX, trainY, epochs=60, batch_size=7, validation_data=(testX, testY), verbose=1)

        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))


    def __call__(self, input):
        return self.categorize(input)


    def categorize(self, input):
        result = None
        return result


# For local testing
def test():
    import pickle
    classifier = CLASSIFIER()

    # Read from file
    with open('training_data.pickle', 'rb') as f:
        training_data = pickle.load(f)

    no_classes = 1
    classifier.train(no_classes, training_data)


if __name__ == "__main__":
    test()
