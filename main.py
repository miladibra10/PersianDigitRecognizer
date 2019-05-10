import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', 'batch', help='batch size')
parser.add_argument('-d', 'dropout', help='dropout rate')
parser.add_argument('-t', 'train', help='training data')
parser.add_argument('-s', 'test', help='test data')
parser.add_argument('-v', 'validation', help='validation split rate')
parser.add_argument('-e', 'epoch', help='epochs')

args = parser.parse_args()

# Default values
batch_size = 1
dropout = 0
train_path = './DigitDB/Train 60000.cdb'
test_path = './DigitDB/Test 20000.cdb'
validation_split_rate = 0
epoch = 1


# Parsing Arguments
if args.batch:
    batch_size = int(args.batch)
if args.dropout:
    dropout = float(args.dropout)
if args.train:
    train_path = args.train
if args.test:
    test_path = args.test
if args.validation:
    validation_split_rate = float(args.validation)
if args.epoch:
    epoch = int(epoch)


model = get_model(dropout)

