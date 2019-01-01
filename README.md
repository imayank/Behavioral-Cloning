# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The strategy was to use the architecture from the nVidia paper. As the architecture was successfully used to train a car to successfully drive in real conditions, I thought it can be used for this project too. The paper by nVidia does not describe if any activation function was used or if something was used to reduce the overfitting of the model. So one of the thing was to update the architecture with non linear activation function and some strategy to reduce the overfitting. Second thing was to collect the appropriate data so the model can perform well on the data. The procedure to collect the the training data is described in the next section.

##### Splitting into training and validation set
Before trying various changes to the model, some strategy was needed to evaluate the model. So, to evaluate the model the data collected was divided into training and validation set. The validation mean squared error (mse) and training mse was used to evaluate how well the model was performing and also to ensure that the model was not overfitting.(*model.py*, line:150). Whenever I found that validation mse is greater than training mse it meant the model has started to overfit the data.

##### Using `fit_generator()` function of keras
After splitting the data into training and validation set, the next step is to train on the model. At first I was reading all the images and was training using `.fit()` function of keras. But  it takes alot of memory and secondly the training process was really slow. To counter this I used `fit_generator()` function of the keras (*model.py*, line:164-168). The `fit_generator()` function trains on batches of data and uses a python generator (*model.py*, line: 82-106) which generated data batches for the function. Using `fit_generator()` helped in saving memory and also speed up the training process. The generators are used for both the training data and validation data (*model.py*, line:153-154).

##### Using Undersampling (unsucessfull):
The first track which is used for recording the data has long run of straight or almost straight roads, as a result of this most of the collected data will record low or zeero steering angle values. So I thought that this will make the model biased towards driving more often in straight lines. Hence I decided to undersample such data. The code for undersampling the data is located in *model.py*, at line:49-71. I was able to achieve low validation accuracy with the final model but while testing on the track the results were not satisfactory. I would definitely try undersampling or try some ither straegy to undersample the data, but for now I failed to use the undersampling to produce sucessful results.

##### Model architectures: 
I started with the model presented in the nVidia paper. Firstly I used the architecture as it is without any activation function and any regularization technique. I trained the model with data recorded and achieved low mse of about 0.1 on validation set. I trained it about 3 epochs after that the mse does not decrease much and also the model starts to overfit the data. I thought it was good. So, I decided to test on the simulator. But I found while driving it was not performing really well and was ultimately throwing it out of the track. I thought maybe it needs more data around the turns. I added more data but it didn't helped much. The model was also not overfitting the data. So at this point I decided to update my architecture.

I decided to add non-linearity to my aarchitecture. I added **Exponential linear unit (ELU)** to introduce non linearity. I used *ELU* as activation function in the present architecture because they tend to converge faster and give more accurate results as compared to other activation functions (Researched this on the internet). So I added *ELU* activation unit after every convolutional layer and every fully connected layer except for the last  fully connected layer. And tried retraining the model. Again I achieved low  validation accuracy, but while testing it was not producing good results.

It was at this point of time I added **L2 regularization** to my model. I thought maybe my model was still overfitting on data as it was huge amount of data, so adding regularization can not damage my model. So, I added regularization to every convolutional layer and every Dense layer except for the last one. I played with the regularization parameter value, and with the value of 0.0001 my car was driving through the track.  

#### 2. Final Model Architecture

The final model architecture (*model.py* lines 110-137) consisted of a convolution neural network with the following layers and layer sizes:

Layer                |        Description
-------------------- | --------------------
Input                | 160x320 normalized RGB image
Cropping2D           | layer to crop the image from top and bottom (20 pixels each)
Convolutional 5x5    | 24 filters with an stride of 2 and VALID padding. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Convolutional 5x5    | 36 filters with an stride of 2 and VALID padding. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Convolutional 5x5    | 48 filters with an stride of 2 and VALID padding. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Convolutional 3x3    | 64 filters with an stride of 1 and VALID padding. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Convolutional 3x3    | 64 filters with an stride of 1 and VALID padding. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Flatten layer        | layer to flatten the image matrix
Dense layer, size:100| Fully connected layer of size 100. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Dense layer, size:50 | Fully connected layer of size 50. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Dense layer, size:10 | Fully connected layer of size 10. Uses an L2 regularizer for kernel weights with value of 0.0001
elu Activation       | ELU activation layer
Dense layer, size:1  | Fully connected layer of size 1.



#### 3. Creation of the Training Set & Training Process

I already have described in the section above how the dataset was collected. I present the process here with exaple images:

For capturing the good driving behaviour following steps were taken:

* **Central lane driving:** The car was driven through the first track keeping the car at the center of the road. The central lane driving data was recorded for two laps. The example images are given below:

central image 1  | central image 2
-----------|-----------
![alt text][image1] | ![alt text][image2]

* **Left side driving:** The car was driven through the first track keeping the car near the left yellow line of the road. The left side driving data was recorded for 2 laps. The example images are given below:

left side driving image 1  | left side driving image 2
-----------|-----------
![alt text][image3] | ![alt text][image4]

* **Right side driving:** The car was driven through the first track keeping the car near the right yellow line of the road. The right side driving data was recorded for 2 laps. The example images are given below:

right side driving image 1  | right side driving image 2
-----------|-----------
![alt text][image5] | ![alt text][image6]

* **Second Track:** To make model more generalized, one lap of data was also recorded from the second track. The example images from second track arepresented below:

second track image 1  | second track image 2
-----------|-----------
![alt text][image7] | ![alt text][image8]

In addition to good driving behaviour, recovery driving behaviour were also recorded. This was specially needed when car moves over the bridge portion of the track and near soil land area of the track. This is also useful when there is a turn in the track

* **Recovery driving:** Some amount of recovery driving was also recorded from the left and right side of the road. This was specially required for the turns and for the bridge portion of the first track. Some example image is presented:

recovery driving image 1  | recovery driving image 2
-----------|-----------
![alt text][image9] | ![alt text][image10]

recovery driving image 3  | recovery driving image 4
-----------|-----------
![alt text][image11] | ![alt text][image12]

recovery driving image 5  | recovery driving image 6
-----------|-----------
![alt text][image17] | ![alt text][image18]


* **Reverse lap:**
Instead of augmenting image data by flipping the images, I actually drive the Track 1 in reverse direction. This will also help to generalize the model. some example image for reverse lap driving is presented below:

reverse lap image 1  | reverse lap image 2
-----------|-----------
![alt text][image13] | ![alt text][image14]

* **Left and right cameras:**  Left and right camera images were also used for training the model. It also helps the model in recovering the car towards the center if it goes left and right. For the image captured from left camera a small value was added to  the steering angle value of central camera so as the car pull towards right and similarly a small negative value was added the steering angle value of central camera to obtaine steering angle for right camera image so that it pull towards right. (*model.py*, line: 25-30). Some example left and right camera images are given below:

left camera image   | right camera image 
-----------|-----------
![alt text][image15] | ![alt text][image16]

After the collection process, I had 61968 number of images. Each image is normalized at the time of generating batches of data for training. (*model.py*, line: 100). The respective change is also made in the file *drive.py* (line: 63)

I split the data into training and validation set. 20% of the data is used for validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 after 3 epochs the validation mse doesn't decreased much also it started overfitting, and  I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 4. Results

The result video is recorded for both the tracks. The car is successfully able to navigate through first track and it almost completes the second track also. Which is a good thing and that also shows the model doesn't overfits the data, as it is primarily trained on the data from the first track.

Video for first track : track_1.mp4
----
Video for second track : track_2.mp4
