import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import shutil


file_list = []
class_list = []

DATADIR = "data"

CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
# The size of
IMG_SIZE = 150
# Checking or

TRAIN_DATA_DIR = "training_data"
TEST_DATA_DIR = "testing_data"

#Percentage of data going to training
TRAIN_DATA_PERCENTAGE = 0.8

#Separate data into training data and testing data
for category in CATEGORIES :
    path = os.path.join(DATADIR, category)
    train_directory = os.path.join(TRAIN_DATA_DIR, category)
    test_directory = os.path.join(TEST_DATA_DIR, category)
    image_directories = os.listdir(path)
    random.shuffle(image_directories)
    index = 0
    while index < (0.8) * len(image_directories):
        source = os.path.join(path, image_directories[index])
        destination = os.path.join(train_directory, image_directories[index])
        shutil.move(source, destination)
        index += 1
    while index < len(image_directories):
        source = os.path.join(path, image_directories[index])
        destination = os.path.join(test_directory, image_directories[index])
        shutil.move(source, destination)
        index += 1



for category in CATEGORIES :
    path = os.path.join(TRAIN_DATA_DIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []

def create_training_data():
    for category in CATEGORIES :
        path = os.path.join(TRAIN_DATA_DIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)


X = [] #features
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
