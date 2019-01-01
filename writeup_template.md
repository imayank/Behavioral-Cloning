# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./example/central_driving_1.jpg "Central 1"
[image2]: ./example/central_driving_2.jpg "Central 2"
[image3]: ./example/left_side_1.jpg "left 1"
[image4]: ./example/left_side_2.jpg "left 2"
[image5]: ./example/right_side_1.jpg "right 1"
[image6]: ./example/right_side_2.jpg "right 2"
[image7]: ./example/second_track_1.jpg "second 1"
[image8]: ./example/second_track_2.jpg "second 2"
[image9]: ./example/recovery_1.jpg "recovery 1"
[image10]: ./example/recovery_2.jpg "recovery 2"
[image11]: ./example/recovery_3.jpg "recovery 3"
[image12]: ./example/recovery_4.jpg "recovery 4"
[image13]: ./example/reverse_1.jpg "reverse 1"
[image14]: ./example/reverse_2.jpg "reverse 2"
[image15]: ./example/left.jpg "left"
[image16]: ./example/right.jpg "right"
[image17]: ./example/recovery_5.jpg "recovery 5"
[image18]: ./example/recovery_6.jpg "recovery 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_best.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_best.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture is shown in thte following table:

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

The architecture used is one from the nVidia paper.The architecture uses convolutional layers with filters of size 5x5 and 3x3 of depth in between 24 and 64. The architecture also uses fully connected layer of size 100,50 and 10. The last layer in the model is a Dense layer of size 1 to produce the steering angle output. The paper does not specifies anything about the what kind of activation functions were used or what was done for overcoming the regularization. At first I used the architecture specified iin the paper as it is, but it didn't produced satiafactory results. So I added elu activation to add some non linearity in the model and L2 regularization for avoiding overfitting of the model. Dropout layer was also tried to deal with overfitting, but l2 regularization produced best results.

The images which are fed to the model are normalized at the time of generating the batches using an python generator (*model.py*, line: 100). The respective change is also made in the file *drive.py* (line: 63)

#### 2. Attempts to reduce overfitting in the model

Following steps were taken to reduce the overfittiing of the data:
* L2 regularization was used in the convoltional and fully connected layer to penalize the kernel weights so that they do not grow out of bounds and overfit the training data. (*model.py*, lines:110-133)
* The data was split in training and validtion sets to check whether or not the model was overfitting the data (*model.py*, line: 150 ). The number of epochs are set on the basis of training mean squared error (mse) and validation mse. If the validation mse was increasing while training mse was decreasing, then the model was probably started to overfitting the model and it is better to stop the training at earlier epoch.
* To model was tested using the simulator by ensuring that the car remains in the track.
* I used data from both the tracks to keep the model more general.

#### 3. Model parameter tuning

A batch size of 128 was used. I didn't tuned the batch size used in the training the model. *Adam optimizer* was used as an optimizer, so that I didn't need to tune for learning rate of the model. (*model.py*, line: 161-168) 

#### 4. Appropriate training data

All the data used for training the self driving car was generated using the *Udacity self driving car simulator*. The data was recorded so as to keep thte car on the track. The data was recorded in the following manner:

* **Central lane driving:** The car was driven through the first track keeping the car at the center of the road. The central lane driving data was recorded for two laps.
* **Left side driving:** The car was driven through the first track keeping the car near the left yellow line of the road. The left side driving data was recorded for 2 laps.
* **Right side driving:** The car was driven through the first track keeping the car near the right yellow line of the road. The right side driving data was recorded for 2 laps.
* **Recovery driving:** Some amount of recovery driving was also recorded from the left and right side of the road. This was specially required for the turns and for the bridge portion of the first track.
* **Second Track:** To make model more generalized, one lap of data was also recorded from the second track.
* **Left and right cameras:**  Left and right camera images were also used for training the model.

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