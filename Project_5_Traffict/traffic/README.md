Traffic.py is a python implementation of an artificial intelligence that 
is capable of reading traffic signs and classify them in a series of 
pre-established categories.

It does so using tensorflow, from now on ts, a library designed by google that implements
many different ML models at your disposal.

It also uses opencv-python, from now on cv2, library in order to read the images and convert
them to numbers.



The proccess of doing so is:

1 - Go folder by folder openning the images and converting them to a numpy array
    of 3 dimensions: (img_width, img_height, pixel_values)

2 - Change the size of each image to get them to be equally large. This allows
    the AI to have the same number of inputs. It does so using an interpolation
    algorithm implemented in the cv2 library.

3 - Split randomly the sample of images and labels for each one, with a test_size of 40%

4 - Create the sequantial model, first approach:
    ·   Convolutional 2D layer with 32 kernels of 3x3, activation = 'relu'
    ·   MaxPooling 2D of 2,2
    ·   Flatten layer
    ·   Hidden layer with 10 nodes, activation = 'relu'
    ·   Dropout layer of 30%.
    ·   Hidden layer with 10 nodes, activation = 'relu'
    ·   Dropout layer of 30%.
    ·   Hidden layer with 10 nodes, activation = 'relu'
    ·   Dropout layer of 30%.
    ·   Output layer of 43 nodes, activation='softmax'

    Results -> Around 55% accuracy.

5 - To improve this model there is a list of things we can do:

    PARAMETERS OF THE SAME MODEL
    ·   Test number of epochs.
    ·   Test the batch size.
    ·   Test number of layers.
    ·   Test number of nodes.
    ·   Test variation of number of nodes between layers:
        ·   Increasing (having more nodes each layer)
        ·   Decreasing (having fewer nodes each layer)
        ·   Interleaved (having high amount and low amount every other layer)
        ·   Equal (having the same amount of nodes per layer)
    ·   Percentage of dropout, altought this can lead to overfitting.

    PARAMETERS OF THE DATA EXTRACTION
    ·   Test size of kernel
    ·   Test number of kernels tried.
    ·   Test number of convolutionals applied.
    ·   Test size of max-pooling matrix.
    ·   Test size of images.

    CHANGE PARAMETERS OF HOW THE MODEL WORKS
    ·   Test different activation methods.

    Many of these parameters have a clear impact on the accuracy of the models. 
    However, we also want to take into consideration the time spent doing so.
    So we are going to measure a ratio of both, which we will call efficiency, 
    being:
    efficiency = ( accurary / ( x * time ) )
    The parameter x will be useful for trying to weight up or down the time parameter 
    when tuning the efficiency to reach for desired results.

    This tests will be done using the file test_model.py (first created in jupyter notebooks,
    file test_model.ipynb)

    To start testing all these parameters we will test them separately, assigning a fixed value
    for the rest of them, so we can find what works best and test them combined for a more
    final accurate testing.

    We will use the same data split for testing and training for the whole iteration of 
    parameters so we can compare equally. But for the purpose of finding the best solution
    overall, we will run the whole thing 3 times with 3 different data splits, and then average
    them.

    From a first batch of data iterating between batch size and epochs. We find the higher
    the batch_size the more accurate and more time efficient. Also, the efficiency decreases
    with higher epochs in training. You can see the results in the 1_batch_epochs.xlsx file.

    After a second batch iterating the combination of nodes and layers, we conclude that more layers
    does not mean better accuracy, on the contrary, it actually lowers the accuracy. Also, we can not
    see any relevant effect of the disposition of the nodes between the layers, meaning having them
    in an increasing, decreasing, interleaved or equally distributed does not affect the outcome.
    





