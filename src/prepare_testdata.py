import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

DATADIR = "../data/test_data"

CATEGORIES = ["target", "no_target"]

training_data = []

IMG_SIZE = 80

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  # ziska index  (0 or a 1). 0 = target 1 = no_target

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # normalizacia - velkosti
                training_data.append([new_array, class_num])
            except: 
                pass

create_training_data()
print(len(training_data))

# priprava modelu
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

# ulozenie dat aby nemuseli byt stale pocitane
pickle_out = open("test_data.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

# nacitanie dat
pickle_out = open("test_label.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()
