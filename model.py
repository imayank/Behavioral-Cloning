import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Conv2D, MaxPooling2D, Activation, BatchNormalization,Dropout


""" Location of the driving_log.csv file and images generated using
Udacity Car Simulator in training mode."""
base_path = "../data/recording/"
base_path_img = "../data/recording/IMG/"

"""Reading in the drving_log file"""
data = pd.read_csv(base_path + "driving_log.csv")


""" This function takes the data frame and merge the three columns 
for the ceter, left , right images into a single column with respective
target steering angle value in the second column."""
def expanding_data(data):
    X_center = data.loc[:,'center']  ## The central camera image
    y_center = data.loc[:,'target']  ## Respective value for steering
    X_left = data.loc[:,'left']      ## The image from left camera
    y_left = y_center + 0.3          ## To steer a bit right add a positive value
    X_right = data.loc[:,'right']    ## The image from the right camera
    y_right = y_center - 0.3         ## To steer a bit left add a negative value
    
    """ Three data frames for central, left, right camera data, each with
    two columns - image location and target value for steering"""
    center_data = pd.concat([X_center,y_center],axis=1,ignore_index=True) 
    left_data = pd.concat([X_left,y_left],axis=1,ignore_index=True)
    right_data = pd.concat([X_right,y_right],axis=1,ignore_index=True)
    
    """Merging the data frames"""
    merged_data = pd.concat([center_data,left_data,right_data],axis=0,ignore_index=True) 
    merged_data.columns=['path','target']
    
    return merged_data
    

""" The function takes as input a data frame and returns a data frame with undersampled
data for some target steering values. The track in the simulator has long almost straight sections,
therefore the data has a large number of observations having low steering angles. Due to this the model may
be biased towards drving straight. The data for such low angle value are undersampled."""
def undersampling(merged_data):
    out = pd.cut(list(merged_data['target']),30,labels=False)  ## divide the steering values in 30 eqaully sized bins
    bins, counts = np.unique(out, return_counts=True) ## count the unique bins and number of values in each bin
    avg_counts = np.mean(counts)  ## average number of values in bins
    target_counts = int(np.percentile(counts,75))  ## the count to which the value will be undersampled -- 75th percentile
    indices = np.where(counts>avg_counts)  ## indices where the counts in the bin is greater than average counts
    target_bins = bins[indices]  ## bins corresponding to the above indices
    
    target_indices = [] ## list holding the undersampled data points
    total_indices = list(range(len(out)))  ## Complete list of indices of the data
    remaining_indices = total_indices      ## list containing the indices remaining after the values in undersampled bins are removed,initialized to the total_indices
    
    ### iterating through bins having value counts greater than avg_counts and undersampling from the those bins
    for value in target_bins:
        bin_ind = list(np.where(out == value)[0])  ## selecting data points in the bin being iterated
        remaining_indices = list(set(remaining_indices) - set(bin_ind))  ## remove the corresponding indices
        random_indices = list(np.random.choice(bin_ind,target_counts, replace=False)) ## randomly selecting 'target_counts' data points from the selected data points
        target_indices.extend(random_indices) ## adding undersampled indices to the list
        
    undersampled_indices = np.concatenate([target_indices,remaining_indices]) ## concatenating the remaining indices with undersampled indices
    undersampled_data = merged_data.loc[undersampled_indices] ## selecting the data points from the data frame
    
    return undersampled_data ## returning the undersampled data
 
    
"""Function that reset the index and adds an "ID" columns to the data frame input """
def reset_and_add(undersampled_data):
    undersampled_data = undersampled_data.reset_index()
    undersampled_data["ID"] = list(range(len(undersampled_data)))
    return undersampled_data

""" The function is a python data generator for producing batches
of data of size = batch_size to be used in keras fit_generator function"""
def dataGenerator(data, batch_size,base_path_img):
    ids = data['ID'].values  ## selecting all the IDs 
    #print(ids)
    num = len(ids) ## length of the data frame
    #indices = np.arange(len(ids))
    np.random.seed(42)
    while True:
        #indices = shuffle(indices)
        np.random.shuffle(ids)  ## shuffling the data
        for offset in range(0,num,batch_size):
            batch = ids[offset:offset+batch_size] ## selectiing a batch
            images = [] ## list holding the batch images
            target = [] ## list holding the steering values corresponding to thte above list
            ## creating a batch of data
            for batch_id in batch:
                img_path = data.loc[batch_id,'path']
                img_name = img_path.split('\\')[-1]
                new_path = base_path_img + img_name
                images.append(((mpimg.imread(new_path))/255)-0.5)
                target.append(data.loc[batch_id,'target'])
                
            images = np.array(images)
            target = np.array(target)
            
            yield images, target ## returning a batch


""" Function that creates a model as given in the NVIDIA research paper"""
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


### Tried undersampling the data, but results were not satisfactory, so end up using complete data for training
"""undersampled_data = undersampling(data)
undersampled_data = expanding_data(undersampled_data)
undersampled_data = reset_and_add(undersampled_data)"""

### using complete data
undersampled_data = expanding_data(data)
undersampled_data = reset_and_add(undersampled_data)

### dividing the data into training and validation sets 
train_data, validation_data = train_test_split(undersampled_data,test_size=0.2,random_state=42)

#create data generators for training and validation with batch size of 128
train_generator = dataGenerator(train_data, 128,base_path_img)
valid_generator = dataGenerator(validation_data,128, base_path_img)

""" creating a model"""
model = model_nvidia_updated()


## Compiling the model using Adam optimizer and mean squared error as loss function
model.compile(loss='mse',optimizer='adam')

## training the model using fit_generator, batch size = 128
model.fit_generator(generator=train_generator,
                    steps_per_epoch = (len(train_data)//128)+1,
                    validation_data=valid_generator,
                    validation_steps = (len(validation_data)//128)+1,
                    epochs = 3)
## saving the model
model.save('model_new.h5')
