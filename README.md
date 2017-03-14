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

### Step 1: Compute Histogram of Oriented Gradients

HOG stands for “Histogram of Oriented Gradients”. Basically, it divides an image in several pieces. For each piece, it calculates the gradient of variation in a given number of orientations. Example of HOG detector — the idea is the image on the right to capture the essence of original image HOG will compute the gradients from blocks of cells. Then, a histogram is constructed with these gradient values.

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features
```

### Step 2: Extract HOG features and build training datasets
```
def extract_features(imgs, cspace='RGB', orient=8,
                        pix_per_cell=8, cell_per_block=4, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)

    # Return list of feature vectors
    return features
```

### Step 3: Train a classifier to identify images of vehicles.

After extracting features from all data, I used a Support Vector Machine Classifier (SVC), with linear kernel, based on function SVM from scikit-learn. In the end, I used color space YCrCb, all channels.

Before training the data, the data was normalized using StandardScaler() from sklearn.preprocessing.

Then these normalized data were splitted into train and test sets.

- **HOG parameter**: orient = 8, pix_per_cell = 8, cell_per_block = 2

```
svc.fit(X_train, y_train)
```

### Step 4: Identify vehicles within images of highway driving.
HOG Subsampling is used to have a efficient way to identifying hog feature in one go. After trying several different scale for different window sizes, I realised using only one scale would be a lot more effective as it allows less false positives.

```
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    boxes =[]

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale),
        np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)).reshape(1, -1)

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell     

            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features)

            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),
                              (0,0,255),6)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return draw_img, boxes
```

### Step 5: Drawing Bounding Rectangles

For this step I defined a function `draw_boxes` which takes as input a list of bounding rectangle coordinates and uses the OpenCV function `cv2.rectangle()` to draw the bounding rectangles on an image.

### Step 6: Track images across frames in a video stream.

For the video implementation, I have used a weighted averaged heatmap to reduce false postive and increase robustness of the pipeline. Specificlly, I do a weight average of the 5 frame, with high importance of the most recent frame, and used that as the heatmap instead.



### Discussion

The hard part of the project is the elimination of false positives. The pipeline as it is still falsely identify road/tree/etc as a car. The most efficient way to improve this model is to use Deep learning like SSD or training a more robust identifier (99%).

Additionally, the pipeline is running very slowly, at serveral seconds per frame. Moving froward, it would be needed to process the image in real time using Deep learning or pruning  the number of features and reducing the number of windows searched.
