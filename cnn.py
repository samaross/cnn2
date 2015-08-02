import numpy as np

def cnn_convolve(filter_dim,num_filters,images,W,b):
    #cnnConvolve Returns the convolution of the features given by W and b with
    #the given images
    #
    # Parameters:
    #  filterDim - filter (feature) dimension
    #  numFilters - number of feature maps
    #  images - large images to convolve with, matrix in the form
    #           images(r, c, image number)
    #  W, b - W, b for features from the sparse autoencoder
    #         W is of shape (filterDim,filterDim,numFilters)
    #         b is of shape (numFilters,1)
    #
    # Returns:
    #  convolvedFeatures - matrix of convolved features in the form
    #                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    num_images = images.shape[2]
    image_dim = images.shape[1]
    conv_dim = image_dim - filter_dim + 1

    convolved_features = np.zeros((conv_dim,conv_dim,num_filters,num_images))
    
    # Instructions:
    #   Convolve every filter with every image here to produce the 
    #   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures x numImages
    #   matrix convolvedFeatures, such that 
    #   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
    #   value of the convolved featureNum feature for the imageNum image over
    #   the region (imageRow, imageCol) to (imageRow + filterDim - 1, imageCol + filterDim - 1)
    #
    # Expected running times: 
    #   Convolving with 100 images should take less than 30 seconds 
    #   Convolving with 5000 images should take around 2 minutes
    #   (So to save time when testing, you should convolve with less images, as
    #   described earlier)

    for image_num in range(0,num_images):
        for filter_num in range(0,num_filters):
            convolved_image = np.zeros((conv_dim,conv_dim))
            
            ### YOUR CODE HERE ###
            # Convolve "filter" with "im", adding the result to convolvedImage
            # be sure to do a 'valid' convolution

            ### YOUR CODE HERE %%%
    
            # Add the bias unit
            # Then, apply the sigmoid function to get the hidden activation

            ### YOUR CODE HERE %%%
            convolved_features[:, :, filter_num, image_num] = convolved_image
    
    return convolved_features


def cnn_pool(pool_dim,convolved_features):
    #cnnPool Pools the given convolved features
    #
    # Parameters:
    #  poolDim - dimension of pooling region
    #  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
    #                     convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
    #
    # Returns:
    #  pooledFeatures - matrix of pooled features in the form
    #                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
    #     
    
    num_images = convolved_features.shape[3]
    num_filters = convolved_features.shape[2]
    convolved_dim = convolved_features.shape[0]
    
    ratio = convolved_dim/pool_dim
    pooled_features = np.zeros((ratio,ratio,num_filters,num_images))
    
    # Instructions:
    #   Now pool the convolved features in regions of poolDim x poolDim,
    #   to obtain the 
    #   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures x numImages 
    #   matrix pooledFeatures, such that
    #   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
    #   value of the featureNum feature for the imageNum image pooled over the
    #   corresponding (poolRow, poolCol) pooling region. 
    #   
    #   Use mean pooling here.

    ### YOUR CODE HERE ###
    
    return pooled_features
