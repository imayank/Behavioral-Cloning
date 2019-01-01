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

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
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
* 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
