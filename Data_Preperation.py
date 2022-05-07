# Importing the libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL
import pathlib
import cv2
from tensorflow.keras import layers
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import csv
import numpy as np
import pandas as pd
import os
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from IPython.display import Image, display
from sklearn import metrics
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#Importing the dataset
data_train = r"C:\Users\giorg\Jupyter notebooks\Artificial Neural Networks_CW\BrainTumor\Training"
data_test = r"C:\Users\giorg\Jupyter notebooks\Artificial Neural Networks_CW\BrainTumor\Testing"

path_train = pathlib.Path(data_train)
path_test = pathlib.Path(data_test)

# map the training path folders to the particular category
categories = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]

for category in categories:
    path_tr = os.path.join(path_train,category)
    
    # read the image as an array
    for img in os.listdir(path_tr):
        img_array_tr = cv2.imread(os.path.join(path_tr,img)) 
        break
    break
    
num = []
for category in categories:
    path = r"C:/Users/giorg/Jupyter notebooks/Artificial Neural Networks_CW/BrainTumor/Training/{0}/".format(category)
    folder_data = os.listdir(path)
    k = 0
    print('\n', category.upper())
    for image_path in folder_data:
        if k < 5:
            display(Image(path+image_path))
        k = k+1
    num.append(k)
    
# just some exploratory plotting    
plt.figure(figsize = (8,8))
plt.bar(categories, num)
plt.title('NUMBER OF IMAGES CONTAINED IN EACH CLASS')
plt.xlabel('classes')
plt.ylabel('count')
plt.show()

# convert all our images to have the same size (some might had different sizes)
img_size = 128

array1 = cv2.resize(img_array_tr,(img_size, img_size)) # resizing all the images

# create the training dataset
train_tumor = []
for i in categories:
    train_path = os.path.join(data_train,i)
    tag = categories.index(i)
    for img in os.listdir(train_path):
        try:
            image_arr = cv2.imread(os.path.join(train_path , img), cv2.IMREAD_GRAYSCALE)
            new_image_array = cv2.resize(image_arr, (img_size,img_size))
            train_tumor.append([new_image_array , tag]) # so the train tumor will consist of the images and the categories 
        except Exception as e:
            pass
          
X_train = []
y_train = []
for i,j in train_tumor:
    X_train.append(i) # i is basically the images
    y_train.append(j) # is basically the categories
    
X_train = np.array(X_train).reshape(-1,img_size,img_size) #transform the image into array so as the computer can read them and reshape it
print(X_train.shape)                                      # the -1 indicates that we now have a 1 dimensional array

X_train = X_train/255.0 # we devide with 255.0 because the pixel intensity lies between 0 - 255 for mathematical simplification
X_train = X_train.reshape(-1,128,128,1)

#create the test dataset
test_tumor = []
for i in categories:
    test_path = os.path.join(data_test,i)
    tag = categories.index(i)
    for img in os.listdir(test_path):
        try:
            image_arr = cv2.imread(os.path.join(test_path , img), cv2.IMREAD_GRAYSCALE)
            new_image_array = cv2.resize(image_arr, (img_size,img_size))
            test_tumor.append([new_image_array , tag]) # so the train tumor will consist of the images and the categories 
        except Exception as e:
            pass
          
X_test = []
y_test = []
for i,j in test_tumor:
    X_test.append(i) # i is basically the images
    y_test.append(j) # is basically the categories
    
X_test = np.array(X_test).reshape(-1,img_size,img_size) #transform the image into array so as the computer can read them and reshape it
print(X_test.shape)                                      # the -1 indicates that we now have a 1 dimensional array

X_test = X_test/255.0 # we devide with 255.0 because the pixel intensity lies between 0 - 255 for mathematical simplification
X_test = X_test.reshape(-1,128,128,1)

# convert the brain tumor categories into numbers using one-hot-encoding
y_train = to_categorical(y_train, num_classes = 4)
y_test = to_categorical(y_test, num_classes = 4)

# our dataset is already split into train-test, so we just need to also use some data for validation
X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)
