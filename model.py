from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import to_categorical
from HodaDatasetReader import read_hoda_dataset


def get_model(dropout):
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(32 * 32,)))
    model.add(Dense(30, activation='relu'))
    if dropout < 1:
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_data(train_path, test_path):
    train_images, train_labels = read_hoda_dataset(train_path)
    test_images, test_labels = read_hoda_dataset(test_path)
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return (train_images, train_labels), (test_images, test_labels)
