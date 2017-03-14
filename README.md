# Vehicle Detection

# Introduction

In Project 5 of the great Udacity Self Driving car nanodegree, the goal is to use computer vision techniques to detect vehicles in a road.

The project is completed in the following stages:
* **Step 1**: Create a function to draw bounding rectangles based on detection of vehicles.
* **Step 2**: Create a function to compute Histogram of Oriented Gradients on image dataset.
* **Step 3**: Extract HOG features from training images and build car and non-car datasets to train a classifier.
* **Step 4**: Train a classifier to identify images of vehicles.
* **Step 5**: Identify vehicles within images of highway driving.
* **Step 6**: Track images across frames in a video stream.

### Step 1: Drawing Bounding Rectangles

For this step I defined a function `draw_boxes` which takes as input a list of bounding rectangle coordinates and uses the OpenCV function `cv2.rectangle()` to draw the bounding rectangles on an image.

### Step 2: Compute Histogram of Oriented Gradients

HOG stands for “Histogram of Oriented Gradients”. Basically, it divides an image in several pieces. For each piece, it calculates the gradient of variation in a given number of orientations. Example of HOG detector — the idea is the image on the right to capture the essence of original image HOG will compute the gradients from blocks of cells. Then, a histogram is constructed with these gradient values.


### Step 3: Extract HOG features and build training datasets


### Step 4: Extract HOG features and build training datasets


### Step 5: Identify vehicles within images of highway driving.

### Step 6: Track images across frames in a video stream.
