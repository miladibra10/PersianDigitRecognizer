from model import get_model, load_data
from matplotlib import pyplot
import argparse
import numpy as np
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', help='batch size', default=1)
parser.add_argument('-d', '--dropout', help='dropout rate', default=1.0)
parser.add_argument('-t', '--train', help='training data', default='./DigitDB/Train 60000.cdb')
parser.add_argument('-s', '--test', help='test data', default='./DigitDB/Test 20000.cdb')
parser.add_argument('-v', '--validation', help='validation split rate', default=0.1)
parser.add_argument('-e', '--epoch', help='epochs', default=10)

args = parser.parse_args()

model = get_model(float(args.dropout))

(train_images, train_labels), (test_images, test_labels) = load_data(args.train_path, args.test_path)

history = model.fit(train_images, train_labels, batch_size=int(args.batch_size), epochs=int(args.epoch), validation_split=float(args.validation_split_rate))

_, train_acc = model.evaluate(train_images, train_labels)
_, test_acc = model.evaluate(test_images, test_labels)

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


predicted = model.predict(test_images, verbose=0)
predicted_classes = model.predict_classes(test_images, verbose=0)

# reduce to 1D array
predicted = np.argmax(predicted, axis=1)

report = classification_report(np.argmax(test_labels, axis=1), predicted)
print(report)
