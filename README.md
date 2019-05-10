# PersianDigitRecognizer
Persian Digit Recognizer using MLP and Keras with HodaDataset.


# Dependencies
+ Python 3.5 or above
+ numpy
+ tensorflow
+ Keras
+ matplotlib
+ sklearn
+ opencv-python (cv2)

# Running the Code
you can run the code with command below:
``` 
main.py [-h] [-b BATCH] [-d DROPOUT] [-t TRAIN] [-s TEST] [-v VALIDATION] [-e EPOCH]
```
all options are optional because of default values. but validation split rate should not be **zero**

+ -h : help for running the program
+ -b : batch size for network learning
+ -d : dropout rate for last layer
+ -t : train data path of `.cdb` file readable by `HodaDatasetReader.py`
+ -s : test data path of `.cdb` file readable by `HodaDatasetReader.py`
+ -v : validation split rate
+ -e : number of epochs

# Example
some examples for executed code and its results are available in `Persian Digit Recognizer.ipynb` file.  



