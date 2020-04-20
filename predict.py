import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "./data/predict_data"

data = []

IMG_SIZE = 80

def ignore_png(img):
    return img.replace('.png','')

def prepare_data():

    path = os.path.join(DATADIR)  

    for img in tqdm(os.listdir(path)):  

        try:
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # normalizacia - velkosti
            img = ignore_png(img)
            data.append([new_array, img])
        except: 
            pass


prepare_data()

X = []
name = []

for features,caption in data:
    X.append(features)
    name.append(caption)

data_for_prediction = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def res(num):
    if num >= 0.5:
        return 1
    else:
        return 0

# load new model
model = keras.models.load_model("model.h5")
print("Loaded model from disk")

####################################################
# make a prediction on dataset
####################################################
predicted = model.predict(data_for_prediction)

predicted = np.argmax(predicted, axis = 1)

# generovanie outputu do suboru
f = open("results.txt", "w")


for i in range(len(predicted)):
    # na prve miesto - nazov suboru 
    f.write(name[i])
    f.write(" ")
    f.write(str(1 - predicted[i]))
    f.write(" ")
    f.write(str(res(1 - predicted[i])))
    f.write("\n")
    
    #print(name[i], 1 - predicted[i], res(1 - predicted[i]))    

f.close()
