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
        data = [([1,2,3,4,5], 1),
                ([1,2,3,4,5], 1),
                ([1,2,2,2,5], 2),
                ([1,2,2,2,5], 2),
                ([1,3,3,3,5], 3)]
        
        datax = [[1,2,3,4,5],
                [1,2,3,4,5],
                [1,2,2,2,5],
                [1,2,2,2,5],
                [1,3,3,3,5]]
        
        datay = [1, 1, 2, 2, 3]
    
        datasetX = [x for x in datax]
        datasetY = [x for x in datay]

        print(f"\nLen x: {len(datasetX)}, Len y: {len(datasetY)}")

        # Create tensorflow dataset & shuffle
        dataset = tf.data.Dataset.from_tensor_slices((datasetX, datasetY))
        #dataset = dataset.shuffle(buffer_size = 1000, seed=123, reshuffle_each_iteration=False)
        #dataset = dataset.batch(16)

        # Split dataset
        train_split = int(round(len(dataset)*0.8))
        print(f"Samples: {len(dataset)}  Split: {train_split}")
        train_ds = dataset.take(train_split)
        test_ds = dataset.skip(train_split)
        
        print("\nprinting elements")
        for element in train_ds:
            print('t')
            print(element)
        

        #exit()

        return train_ds, test_ds


    def _get_model(self, no_classes=5, dropout_rate=0.2):
        model = Sequential()
        model.add(Input(shape=(5,), name="Initial-Input"))
        model.add(Dense(55, activation='relu', kernel_initializer='he_uniform', name="First-Dense"))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        model.add(Dense(no_classes, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Model Summary: {model.summary()}")
        return model


    def train(self, no_classes, data):
        model = self._get_model(no_classes=no_classes, dropout_rate=0.2)
        train_ds, test_ds = self._get_data(data)

        print("len train_ds", len(train_ds))
        print("len test_ds", len(test_ds))

        print("train_ds", train_ds)
        print("test_ds", test_ds)

        # fit model
        print("\nFitting model")
        history = model.fit(train_ds, epochs=100, batch_size=7, validation_data=test_ds, verbose=1)
        #history = model.fit(tf.expand_dims(train_ds, axis=-1), epochs=100, batch_size=7, validation_data=test_ds, verbose=1)

        # evaluate model
        _, acc = model.evaluate(test_ds, verbose=0)
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

    no_classes = 5
    classifier.train(no_classes, training_data)


if __name__ == "__main__":
    test()
