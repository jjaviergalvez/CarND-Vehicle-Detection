import matplotlib.image as mpimg
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import learning_curve


def myimread(img_file):
    """
    Funtion to read an RGB image and avoid the problems with the diference with
    read an image in .png format and .jpg format with matplotlib
    """
    
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img



def data_look(car_list, notcar_list):
    """
    For display some info about the dataset

    """
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    ind = np.random.randint(0, data_dict["n_cars"])
    img = myimread(car_list[ind])
    data_dict["image_shape"] = img.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = img.dtype
    # Return data_dict
    return data_dict



def get_data_fileNames(setName):
    """
    Return the filenames of cars and notcars associate to the setName
    """
    notcars = []
    cars = []

    images = glob.glob('data/'+setName+'/non-vehicles/Extras/*.png')
    for image in images:
        notcars.append(image)        
    images = glob.glob('data/'+setName+'/non-vehicles/GTI/*.png')
    for image in images:
        notcars.append(image)

    images = glob.glob('data/'+setName+'/vehicles/GTI_Far/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('data/'+setName+'/vehicles/GTI_Left/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('data/'+setName+'/vehicles/GTI_MiddleClose/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('data/'+setName+'test/vehicles/GTI_Right/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('data/'+setName+'/vehicles/KITTI_extracted/*.png')
    for image in images:
        cars.append(image)

    return cars, notcars

def convert_color(img, cSpace="RGB"):
    """
    Image color conversion function.

    -----
    Parameters:
    img: an RGB image
    cSpace: the color space to be converted 

    -----
    Returns:
    feature_image: the converted image

    """
    if cSpace != 'RGB':
        if cSpace == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cSpace == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cSpace == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cSpace == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cSpace == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)

    return feature_image    

def bin_spatial(img, size=(32, 32)):
    """
    Function to compute binned color features

    -----
    Parameters:
    img: an image
    size: a tupple with the resized image

    -----
    Returns:
    features: a vector with the binned color features
    """

    features = cv2.resize(img, size).ravel() 

    return features
                        
def color_hist(img, nbins=32):
    """
    Compute the color histogram of an image

    Parameters
    ----------
    img : an image
    n_bins : the number og bins of the histogram

    
    Returns
    -------
    hist_features : the feature vector with the color histogram
    """
    
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    """
    Compute the Histogram Oriented Gradient features of an image

    """
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features


def hog_extract_features(img, orient, pix_per_cell, cell_per_block, hog_channel):
    """
    HOG feature extraction of an image in particular channel or in all channels

    """
    # Call get_hog_features() with vis=False, feature_vec=True
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(get_hog_features(img[:,:,channel], 
                                orient, pix_per_cell, cell_per_block, 
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)        
    else:
        hog_features = get_hog_features(img[:,:,hog_channel], orient, 
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    
    # Return list of feature vectors
    return hog_features


def list_extract_features(imgs_list, cSpace='HSV',
                        spatial_feat=True, spatial_size=(16, 16),
                        hist_feat=True, hist_bins=32,
                        hog_feat=True, orient=6, pix_per_cell=8, cell_per_block=8, channel=1):
    """
    This function take as main input a list of the images file name and extract the features of each image

    """

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs_list:
        file_features = []
        # Read in each one by one
        img = myimread(file)

        # apply color conversion if other than 'RGB'
        img = convert_color(img, cSpace=cSpace)
        
        if spatial_feat==True:
            spatial_features = bin_spatial(img, size=spatial_size)
            file_features.append(spatial_features)
            
        if hist_feat==True:
            hist_features = color_hist(img, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat==True:
            hog_features = hog_extract_features(img, orient, pix_per_cell, cell_per_block, channel)
            file_features.append(hog_features)

        features.append(np.concatenate(file_features))

    return features

def img_extract_features(img, cSpace='HSV',
                        spatial_feat=True, spatial_size=(16, 16),
                        hist_feat=True, hist_bins=32,
                        hog_feat=True, orient=6, pix_per_cell=8, cell_per_block=8, channel=1):
    """
    This function compute the feature extraction of a single image

    """

    features = []

    # apply color conversion if other than 'RGB'
    img = convert_color(img, cSpace=cSpace)
    
    if spatial_feat==True:
        spatial_features = bin_spatial(img, size=spatial_size)
        features.append(spatial_features)
        
    if hist_feat==True:
        hist_features = color_hist(img, nbins=hist_bins)
        features.append(hist_features)

    if hog_feat==True:
        hog_features = hog_extract_features(img, orient, pix_per_cell, cell_per_block, channel)
        features.append(hog_features)

    return np.concatenate(features)


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cells_per_step):
    """
    Single function that can extract features using hog sub-sampling and make predictions

    Parameters
    ----------
    img : an RGB image scaled 0 to 255
    ystart : y pixel position to start searching
    ystop : y pixel position to stop searching
    svc : SVM classifier
    X_scaler : the factor that will be multiplied by 64 to determined the window size
    .
    .
    .
    cells_per_step: determined the overlaping of the window

    Returns
    -------
    windows_list : the windows where the svm predicted to be a car 
    """
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255 # I didn't use this in my training stage
    #img = img.astype(np.float32) # With this, I have a significal diference in prediction
                                  # compare with the ineficient method (search_window()) and
                                  # as the ineffient methos has much better performance I decided
                                  # to ommit this step.
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, 'HSV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
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
    #cells_per_step = 8  # Instead of overlap, define how many cells to step
    #nxsteps = (nxblocks - nblocks_per_window) // cells_per_step 
    #nysteps = (nyblocks - nblocks_per_window) // cells_per_step 
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)   

   
    # to store the windows detected
    windows_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            #subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))  
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                windows_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
                #cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
            """
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)
            win_draw = np.int(window*scale)
            windows_list.append(((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart)))
            """
    
    #return draw_img          
    return windows_list
    

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)): 

    """
    Function that takes an image, start and stop positions in both x and y,
    window size (x and y dimensions), and overlap fraction (for both x and y)
    and return a list of the cordinates of each window

    """


    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    #nx_windows = np.int(xspan/nx_pix_per_step) - 1
    #ny_windows = np.int(yspan/ny_pix_per_step) - 1
    nx_windows = np.int((xspan - xy_window[0])/nx_pix_per_step) + 1
    ny_windows = np.int((yspan - xy_window[1])/ny_pix_per_step) + 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def search_windows(img, windows, clf, scaler, cSpace='HSV',
                spatial_feat=True, spatial_size=(16, 16),
                hist_feat=True, hist_bins=32,
                hog_feat=True, orient=6, pix_per_cell=8, cell_per_block=8, channel=1):
    """
    Function you will pass an image and the list of windows 
    to be searched (mat be the output of slide_windows())

    """

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = img_extract_features(test_img, cSpace,
                                    spatial_feat, spatial_size,
                                    hist_feat, hist_bins,
                                    hog_feat, orient, pix_per_cell, cell_per_block, channel)

        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)
    # Return the image
    return img

def labels2boxes(labels):
    # Iterate through all detected cars
    box_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))

        #add
        box_list.append(bbox)
        
    # Return the image
    return box_list


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).

    Credits to: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.grid()
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training acc")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test acc")
    
    plt.legend(loc="best")
    
    return plt