#**Traffic Sign Recognition by Gaspard Shen** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Pictures/trainingdata_visual.png "Visual"
[image2]: ./Pictures/grayscale.png "Grayscaling"
[image3]: ./Pictures/normalized.png "Normalized"
[image4]: ./Pictures/OpenCV.png "Data Augmentation"
[image5]: ./Pictures/data_aug_visual.png "Visual"
[image6]: ./Pictures/MS_ConvNet.png "MS_ConvNet"

[image7]: ./New_Image/Class0_Img.png "Traffic Sign 1"
[image8]: ./New_Image/Class31_Img.png "Traffic Sign 2"
[image9]: ./New_Image/Class16_Img.png "Traffic Sign 3"
[image10]: ./New_Image/Class3_Img.png "Traffic Sign 4"
[image11]: ./New_Image/Class14_Img.png "Traffic Sign 5"
[image12]: ./New_Image/Class26_Img.png "Traffic Sign 6"
[image13]: ./New_Image/Class18_Img.png "Traffic Sign 7"
[image14]: ./New_Image/Class34_Img.png "Traffic Sign 8"
[image15]: ./New_Image/Class33_Img.png "Traffic Sign 9"
[image16]: ./New_Image/Class17_Img.png "Traffic Sign 10"

[image17]: ./Pictures/newimage.png "New Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Here is a link to my [project code](https://github.com/igaspard/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? 34799
* The size of the validation set is ? 4410
* The size of test set is ? 12630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many of each class of the training data.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. 

As a first step, I decided to convert the images to grayscale because it is simple and can reduce two channel of chroma data.
This can reduce the complexity of our conv network and improve the traing effienecy.
Moreover the reference paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" also mention the grayscale good result.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the light condition is various in our sample and real world.
After normalized image can avoid the too light image and too dark image.
Here is an example of a traffic sign image before and after normalized.

![alt text][image3]

I decided to generate additional data because the training data distribution was imbalance.
The max number of the training data is Speed limit (50km/h), it has 2010 samples, but the min one is Speed limit (20km/h) and only 180 samples. The machine learning need more data to improve it accuracy.
To add more data to the the data set, I used the OpenCV image processing techniques such scale, brightness... etc to apply at the normalized image and increase the data set. 
For the data below 1000, i apply the OpenCV and generate the sample up to 1000.

Here is an example of an original image and an augmented image:

![alt text][image4]

After apply the data augmentation, here is the distribution of data. You can observe more balance data set.

![alt text][image5]

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Apply the model as mention in the reference paper.
![alt text][image6]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding output 400  |
| RELU					|												|
| Flatten & Concat		| conv2 and conv3 output 	|
| Dropout				| 0.5        									| 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an the model based on the LeNet. Using the AdamOptimizer, batch size 128 and 100 EPOCHS. 
Learning rate is 0.0005

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of ? 0.971
* validation set accuracy of ? 0.971
* test set accuracy of ? 0.947
 
Here list the history of my process to archieve accuracy 0.971
As you can see apply the MS ConvNet, tuning parameter and Data Augmentation are all have amount to improvement.
1. LeNet w/ RGB
EPOCH 10 ...
Validation Accuracy = 0.890

2. LeNet w/ GrayScale, a little improve.
EPOCH 10 ...
Validation Accuracy = 0.900

3. Apply Normalizing, look like same as GrayScale
EPOCH 10 ...
Validation Accuracy = 0.897

4. MS ConvNet, achieve 0.94!
EPOCH 10 ...
Validation Accuracy = 0.946

5. Tuning the hyperparameter learning rate and EPOCHS
EPOCH 100 ...
Validation Accuracy = 0.957

6. Data Augmentation
EPOCH 100 ...
Validation Accuracy = 0.971


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image7] ![alt text][image8] ![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12] ![alt text][image13] ![alt text][image14] ![alt text][image15] ![alt text][image16]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

100% accuracy!

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Look like pretty accuracy
![alt text][image17]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


