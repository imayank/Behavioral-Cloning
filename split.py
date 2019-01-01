import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from shutil import copyfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D, Activation
from data_generator import DataGenerator

params = {'dim': (160,320),
          'batch_size': 128,
          'n_channels': 3,
          'shuffle': True}


data = pd.read_csv("driving_log.csv")
#print(data.columns)
X_center = data.loc[:,'center']
y_center = data.loc[:,'target']
X_left = data.loc[:,'left']
y_left = y_center + 0.1
X_right = data.loc[:,'right']
y_right = y_center - 0.1

center_data = pd.concat([X_center,y_center],axis=1,ignore_index=True)
left_data = pd.concat([X_left,y_left],axis=1,ignore_index=True)
right_data = pd.concat([X_right,y_right],axis=1,ignore_index=True)

merged_data = pd.concat([center_data,left_data,right_data],axis=0,ignore_index=True)
merged_data.columns=['path','target']
merged_data=merged_data.reset_index()

merged_data['ID'] = list(range(len(merged_data)))

X_train, X_test, y_train,y_test = train_test_split(merged_data.loc[:,['ID','path']],merged_data.loc[:,'target'],test_size=0.2,random_state=42)

X_train.columns = ['ID','path']
X_test.columns = ['ID','path']
y_train.columns = ['target']
y_test.columns = ['target']

train_data = pd.concat([X_train,y_train],axis=1,ignore_index=True)
test_data = pd.concat([X_test,y_test],axis=1,ignore_index=True)

train_data.columns = ['ID','path','target']
test_data.columns = ['ID','path','target']

#train_data.to_csv("train_data.csv")
#test_data.to_csv("test_data.csv")

partition = {}

partition['train'] = list(train_data.loc[:,'ID'])
partition['test'] = list(test_data.loc[:,'ID'])
labels_train = dict(zip(train_data.loc[:,'ID'],train_data.loc[:,'target']))
labels_test = dict(zip(test_data.loc[:,'ID'],test_data.loc[:,'target']))


training_generator = DataGenerator(partition['train'], labels_train,train_data, **params)
validation_generator = DataGenerator(partition['test'], labels_test,test_data, **params)


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
model.fit_generator(generator=training_generator,
                    steps_per_epoch = training_generator.__len__(),
                    validation_data=validation_generator,
                    validation_steps = validation_generator.__len__(),
                    epochs = 2,
                    use_multiprocessing=True,
                    workers=6)
model.save('model.h5')                                