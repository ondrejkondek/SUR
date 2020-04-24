import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np 
import os
from sklearn.metrics import confusion_matrix


# nastavenie procesora - veronika nebude mozno potrebovat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load new model
model = keras.models.load_model("model.h5")
print("Loaded model from disk")
 

pickle_in = open("test_data.pickle","rb")
test_data = pickle.load(pickle_in)

pickle_in = open("test_label.pickle","rb")
test_label = pickle.load(pickle_in)

####################################################
# testing data
####################################################
predicted = model.predict(test_data)
print(predicted)

predicted = np.argmax(predicted, axis = 1)
print("\n")
print(test_label)
print('\n')
print(confusion_matrix(test_label, predicted))
####################################################
