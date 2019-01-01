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
* L2 regularization was used in the convoltional and fully connected layer to penalize the kernel weights so that they do not grow out of bounds and overfit the training data. (*model.py*, lines:110-133)
* The data was split in training and validtion sets to check whether or not the model was overfitting the data (*model.py*, line: 150 ). The number of epochs are set on the basis of training accuracy and validation accuracy. If the validation accuracy was dropped while training accuracy was increasing, then the model was probably started to overfitting the model and it is better to stop the training at earlier epoch.
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
