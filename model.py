import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D, Activation

def normalize(image):
    return (image/255.0)-0.5
lines = []
images = []
measurements = []

with open('../data/recording/driving_log.csv') as drive_log:
    reader = csv.reader(drive_log)
    for line in reader:
        lines.append(line)

base_path = ('../data/recording/IMG/')
for line in lines:
    central_image_path = line[0]
    left_image_path = line[1]
    right_image_path = line[2]
    central_name = central_image_path.split('\\')[-1]
    left_name = left_image_path.split('\\')[-1]
    right_name = right_image_path.split('\\')[-1]
    current_path_central = base_path + central_name
    current_path_left = base_path + left_name
    current_path_right = base_path + right_name
    central_image = mpimg.imread(current_path_central)
    left_image = mpimg.imread(current_path_left)
    right_image = mpimg.imread(current_path_right)
    central_measure = float(line[3])
    images.extend([central_image,left_image,right_image])
    measurements.extend([central_measure,central_measure+0.1,central_measure-0.1])
                                  
X_train = np.array(images)
y_train = np.array(measurements)
print(X_train.shape,y_train.shape)
                              
model = Sequential()
#model.add(Lambda(normalize,input_shape=(160,320,3)))
model.add(Cropping2D(((20,20),(0,0)),input_shape=(160,320,3)))
model.add(Conv2D(64,3,padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(128,3,padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(256,3,padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.3,shuffle=True,nb_epoch=5)
model.save('model.h5')                                