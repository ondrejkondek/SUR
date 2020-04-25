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

# nastavenie procesora
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

model.compile(loss='binary_crossentropy',
optimizer='adam',
metrics=['accuracy'])

######### TOTO je na mojich preorganizovanych datach
model.fit(X, y, batch_size=16, epochs=10, validation_split=0.2)


######## TOTO je na povodnych datach (vela pokusov)
# bez dropoutu na 1 epoche
#[[ 8  2]
#[13 47]]
# TOTO na batch_size=16, epochs=1, validation_split=0.3
#[[10  0]
#[14 46]]

#model.fit(X, y, batch_size=16, epochs=1, validation_split=0.3)
#########




model.save("model.h5")
print("Saved model to disk")
