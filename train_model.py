import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np 

import os
# nastavenie procesora - veronika nebude mozno potrebovat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

X = X/255.0

y = to_categorical(y)

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(2))
model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',



model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit(X, y, batch_size=16, epochs=15, validation_split=0.2)

#model.predict()


pickle_in = open("test_data.pickle","rb")
test_data = pickle.load(pickle_in)

pickle_in = open("test_label.pickle","rb")
test_label = pickle.load(pickle_in)

from sklearn.metrics import confusion_matrix

predicted = model.predict(test_data)

predicted = np.argmax(predicted, axis = 1)

print(predicted)
print("\n")
print(test_label)
print(confusion_matrix(test_label, predicted))


# import numpy as np
# import keras
# from keras.models import Sequential
# from keras.layers import Activation
# from keras.layers.core import Dense, Flatten
# from keras.optimizers import Adam
# from keras.metrics import categorical_crossentropy
# from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.normalization import BatchNormalization
# from keras.layers.convolutional import Conv2D
# from sklearn.metrics import confusion_matrix
# import itertools
# import matplotlib.pyplot as plt

# train_path = './data_projekt/trainwithaugm2/train/'
# valid_path = './data_projekt/trainwithaugm2/valid/'
# 
# train_batches = ImageDataGenerator().flow_from_directory(directory=train_path, target_size=(50,50),
#     classes=['target', 'non_target'], batch_size=10)
# valid_batches = ImageDataGenerator().flow_from_directory(directory=valid_path, target_size=(50,50),
#     classes=['target', 'non_target'], batch_size=10)
# 
# model = Sequential([
# Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(50,50,3)),
# Flatten(),
# Dense(2, activation='softmax'),
# ])
# 
# model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
# 
# model.fit_generator(generator=train_batches, steps_per_epoch=93, 
# validation_data=valid_batches, validation_steps=12, epochs=3, verbose=2)
