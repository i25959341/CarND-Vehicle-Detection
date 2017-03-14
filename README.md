# Vehicle Detection
<a href="https://imgflip.com/gif/1ld6om"><img src="https://i.imgflip.com/1ld6om.gif" title="made at imgflip.com"/></a>

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

```
$(function(){
  $('div').html('I am a div.');
});
```

### Step 3: Extract HOG features and build training datasets


### Step 4: Train a classifier to identify images of vehicles.

After extracting features from all data, I used a Support Vector Machine Classifier (SVC), with linear kernel, based on function SVM from scikit-learn. In the end, I used color space YCrCb, all channels.

Before training the data, the data was normalized using StandardScaler() from sklearn.preprocessing.

Then these normalized data were splitted into train and test sets.

- **HOG parameter**: orient = 8, pix_per_cell = 8, cell_per_block = 2

### Step 5: Identify vehicles within images of highway driving.
HOG Subsampling is used to have a efficient way to identifying hog feature in one go. After trying several different scale for different window sizes, I realised using only one scale would be a lot more effective as it allows less false positives.

### Step 6: Track images across frames in a video stream.

For the video implementation, I have used a weighted averaged heatmap to reduce false postive and increase robustness of the pipeline. Specificlly, I do a weight average of the 5 frame, with high importance of the most recent frame, and used that as the heatmap instead.



### Discussion

The hard part of the project is the elimination of false positives. The pipeline as it is still falsely identify road/tree/etc as a car. The most efficient way to improve this model is to use Deep learning like SSD or training a more robust identifier (99%).

Additionally, the pipeline is running very slowly, at serveral seconds per frame. Moving froward, it would be needed to process the image in real time using Deep learning or pruning  the number of features and reducing the number of windows searched.
