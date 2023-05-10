# import data processing and visualisation libraries
import numpy as np
import pandas as pd

# import image processing libraries
import cv2
import skimage
from skimage.transform import resize

# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping

import time

print("Packages imported...")
batch_size = 64
imageSize = 64
target_dims = (imageSize, imageSize, 3)
num_classes = 29

train_len = 87000
train_dir = 'archive/asl_alphabet_train/asl_alphabet_train/'

def get_data(folder):
    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len,), dtype=int)
    cnt = 0
    
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28           
            else:
                label = 29
            for image_filename in os.listdir(folder + folderName):
                img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    
                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1
    return X,y
X_init, y_init = get_data(train_dir)
print("Images successfully imported...")

# Create a copy of data
X_data = X_init
y_data = y_init

# Assuming X_data is a numpy array of shape (N, D) and y_data is a numpy array of shape (N,)
# where N is the number of samples and D is the number of features.

num_classes = 29
num_samples_per_class = 215

X_new = []
y_new = []

for c in range(num_classes):
    indices = np.where(y_data == c)[0]
    indices = np.random.choice(indices, size=num_samples_per_class, replace=False)
    X_new.append(X_data[indices])
    y_new.append(y_data[indices])

X_data = np.vstack(X_new)
y_data = np.hstack(y_new)

# Train Test Split
X_train_original, X_test_original, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3,random_state=42,stratify=y_data)

# One hot encoding target data
y_cat_train = to_categorical(y_train,29)
y_cat_test = to_categorical(y_test,29)

# CNN Model
cnn_5 = Sequential()

cnn_5.add(Conv2D(32, (5, 5), input_shape=(64, 64, 3)))
cnn_5.add(BatchNormalization())
cnn_5.add(Activation('relu'))
cnn_5.add(MaxPooling2D((2, 2)))

cnn_5.add(Conv2D(64, (3, 3)))

cnn_5.add(Activation('relu'))
cnn_5.add(MaxPooling2D((2, 2)))

cnn_5.add(Conv2D(64, (3, 3)))

cnn_5.add(Activation('relu'))
cnn_5.add(MaxPooling2D((2, 2)))

cnn_5.add(Flatten())

cnn_5.add(Dense(128, activation='relu'))

cnn_5.add(Dense(29, activation='softmax'))

cnn_5.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_cnn_5 = cnn_5.fit(X_train_original, y_cat_train,
          epochs=40,
          batch_size=64,
          verbose=2,
          validation_data=(X_test_original, y_cat_test))

# Saving model
cnn_5.save("cnn_5.h5")