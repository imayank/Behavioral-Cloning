import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D, Activation, BatchNormalization,Dropout

base_path = "../data/recording/"
base_path_img = "../data/recording/IMG/"

#base_path = "/opt/data/recording/"
#base_path_img = "/opt/data/recording/IMG/"



data = pd.read_csv(base_path + "driving_log.csv")

def expanding_data(data):
    X_center = data.loc[:,'center']
    y_center = data.loc[:,'target']
    X_left = data.loc[:,'left']
    y_left = y_center + 0.3
    X_right = data.loc[:,'right']
    y_right = y_center - 0.3
    
    center_data = pd.concat([X_center,y_center],axis=1,ignore_index=True)
    left_data = pd.concat([X_left,y_left],axis=1,ignore_index=True)
    right_data = pd.concat([X_right,y_right],axis=1,ignore_index=True)
    
    merged_data = pd.concat([center_data,left_data,right_data],axis=0,ignore_index=True)
    merged_data.columns=['path','target']
    
    return merged_data
    

def undersampling(merged_data):
    out = pd.cut(list(merged_data['target']),30,labels=False)
    bins, counts = np.unique(out, return_counts=True)
    avg_counts = np.mean(counts)
    target_counts = int(np.percentile(counts,75))
    indices = np.where(counts>avg_counts)
    target_bins = bins[indices]
    
    target_indices = []
    total_indices = list(range(len(out)))
    remaining_indices = total_indices
    
    for value in target_bins:
        bin_ind = list(np.where(out == value)[0])
        remaining_indices = list(set(remaining_indices) - set(bin_ind))
        random_indices = list(np.random.choice(bin_ind,target_counts, replace=False))
        target_indices.extend(random_indices)
        
    undersampled_indices = np.concatenate([target_indices,remaining_indices])
    undersampled_data = merged_data.loc[undersampled_indices]
    
    return undersampled_data
    
def reset_and_add(undersampled_data):
    undersampled_data = undersampled_data.reset_index()
    undersampled_data["ID"] = list(range(len(undersampled_data)))
    return undersampled_data

def dataGenerator(data, batch_size,base_path_img):
    ids = data['ID'].values
    #print(ids)
    num = len(ids)
    #indices = np.arange(len(ids))
    np.random.seed(42)
    while True:
        #indices = shuffle(indices)
        np.random.shuffle(ids)
        for offset in range(0,num,batch_size):
            batch = ids[offset:offset+batch_size]
            images = []
            target = []
            for batch_id in batch:
                img_path = data.loc[batch_id,'path']
                img_name = img_path.split('\\')[-1]
                new_path = base_path_img + img_name
                images.append(((mpimg.imread(new_path))/255)-0.5)
                target.append(data.loc[batch_id,'target'])
                
            images = np.array(images)
            target = np.array(target)
            
            yield images, target
                
def model_VGG():
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
    model.add(Dropout(0.7))
    model.add(Dense(100))
    model.add(Dropout(0.7))
    model.add(Dense(50))
    model.add(Dropout(0.7))
    model.add(Dense(10))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    
    return model

def model_nvidia_orig():
    model = Sequential()
    model.add(Cropping2D(((20,20),(0,0)),input_shape=(160,320,3)))
    model.add(Conv2D(24,5,strides=(2,2),padding='valid'))
    model.add(Conv2D(36,5,strides=(2,2),padding='valid'))
    model.add(Conv2D(48,5,strides=(2,2),padding='valid'))
    model.add(Conv2D(64,3,strides=(1,1),padding='valid'))
    model.add(Conv2D(64,3,strides=(1,1),padding='valid'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    return model

def model_nvidia_updated():
    model = Sequential()
    model.add(Cropping2D(((20,20),(0,0)),input_shape=(160,320,3)))
    model.add(Conv2D(24,5,strides=(2,2),padding='valid',kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(36,5,strides=(2,2),padding='valid',kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(48,5,strides=(2,2),padding='valid',kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(64,3,strides=(1,1),padding='valid',kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Conv2D(64,3,strides=(1,1),padding='valid',kernel_regularizer=regularizers.l2(0.0001)))
    #model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(100,kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('elu'))
    model.add(Dense(50,kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('elu'))
    model.add(Dense(10,kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('elu'))
    model.add(Dense(1))
    
    return model

train_data, validation_data = train_test_split(data,test_size=0.2,random_state=42)

#merged_data = expanding_data(data)
#undersampled_data = undersampling(merged_data)
#undersampled_data = reset_and_add(undersampled_data)
train_data = expanding_data(train_data)
undersampled_data = undersampling(train_data)
undersampled_data = reset_and_add(undersampled_data)

validation_data = expanding_data(validation_data)
validation_data = reset_and_add(validation_data)
"""
undersampled_data = expanding_data(data)
undersampled_data = reset_and_add(undersampled_data)"""



#print(train_data.columns)
train_generator = dataGenerator(undersampled_data, 128,base_path_img)
valid_generator = dataGenerator(validation_data,128, base_path_img)

#model = model_nvidia_orig()
model = model_nvidia_updated()
#model = model_VGG()
model.compile(loss='mse',optimizer='adam')

model.fit_generator(generator=train_generator,
                    steps_per_epoch = (len(train_data)//128)+1,
                    validation_data=valid_generator,
                    validation_steps = (len(validation_data)//128)+1,
                    epochs = 5)
model.save('model_new.h5')
