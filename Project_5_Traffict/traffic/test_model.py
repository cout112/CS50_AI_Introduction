import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

images_main_folder = 'gtsrb'
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4
X = 1

def load_data(data_dir):
    """
    Loads all the images into a list of numpy arrays
    """
    folders = os.listdir(data_dir)
    dimension = ( IMG_WIDTH, IMG_HEIGHT )
    images = []
    labels = []
    for folder in folders:
        if folder.startswith('.'):
            continue
        image_names = os.listdir(os.path.join(data_dir, folder))
        for image_name in image_names:
            image = cv2.imread(os.path.join(data_dir, folder, image_name))
            image = cv2.resize(image, dimension, interpolation = cv2.INTER_AREA)
            images.append(image)
            labels.append(folder)
    return images, labels

def create_parameters_combination(  combine:bool = False, 
                                    epochs:list = [EPOCHS], 
                                    batch_sizes:list = [25],
                                    layers:list = [1], 
                                    nodes:list = [100], 
                                    node_variations:list = ['equal'], 
                                    dense_activations:list = ['relu'],
                                    dropouts:list = [0.5], 
                                    kernel_sizes:list = [3], 
                                    kernel_tries:list = [32], 
                                    convolutionals:list = [1], 
                                    convolutional_activations:list = ['relu'],
                                    max_pooling_sizes:list = [2] ) -> dict: 
    """
    This creates a set of dictionaries with the parameters for the creation of the model 
    and the training of the same.
    Args:
        combine: Default False, this decides if you make a combination of all parameters
         or just set default for each one and iterate on one at the time. The default 
         value for each parameter is the first item in the list.
        epochs: number of epochs in which to train the mode. Is a list of integers.
        batch_size: List of batch sizes.
        layers: number of Dense layers to apply to the model. List of integers.
        nodes: Total number of nodes for all layers. It will be rounded down if not divisible by 
         number of layers.
        node_variation: Decides the amount of nodes per layer in weights, you can have a descending
         amount per layer, an ascending amount, an interleaved, or an equal amount.
        dense_activations: List with activation equation for hidden layers.
        dropout: percentage of dropout for last layer.
        kernel_size: Sice of kernel matrix.
        kernel_tries: Amount of times a new kernel is tried.
        convolutionals: List with amount of convolutional layers applied.
        convolutional_activations: List with activation methods for the convolutional.
        max_pooling_sizes: List with amount of max_pooling matrix size.
    """

    available_node_variations = [ 'equal',
                                'ascending',
                                'descending',
                                'interleaved']

    for variation in node_variations:
        if not variation in available_node_variations:
            raise Exception(f'node variation not in options: Must choose {available_node_variations}')

    parameters_list = []
    for epoch in epochs:
        first_epoch = epoch == epochs[0]
        for batch_size in batch_sizes:
            first_batch_size = batch_size == batch_sizes[0]
            for layer in layers:
                first_layer = layer == layers[0]
                for node_num in nodes:
                    first_node_num = node_num == nodes[0]
                    for node_variation in node_variations:
                        first_node_variation = node_variation == node_variations[0]
                        for dense_activation in dense_activations:
                            first_dense_activation = dense_activation == dense_activations[0]
                            for dropout in dropouts:
                                first_dropout = dropout == dropouts[0]
                                for kernel_size in kernel_sizes:
                                    first_kernel_size = kernel_size == kernel_sizes[0]
                                    for kernel_try in kernel_tries:
                                        first_kernel_try = kernel_try == kernel_tries[0]
                                        for convolutional in convolutionals:
                                            first_convolutional = convolutional == convolutionals[0]
                                            for convolutional_activation in convolutional_activations:
                                                first_convolutional_activation = convolutional_activation == convolutional_activations[0]
                                                for max_pooling_size in max_pooling_sizes:
                                                    first_max_pooling_size = max_pooling_size == max_pooling_sizes[0]
                                                    # print(f"max_pooling_size:  {max_pooling_size}")
                                                    # print(f"convolutional:  {convolutional}")
                                                    # print(f"kernel_try:  {kernel_try}")
                                                    # print(f"kernel_size:  {kernel_size}")
                                                    # print(f"dropout:  {dropout}")
                                                    # print(f"node_num:  {node_num}")
                                                    # print(f"layer:  {layer}")
                                                    # print(f"batch_size:  {batch_size}")
                                                    # print(f"epoch:  {epoch}")
                                                    parameters_dict = {
                                                        'max_pooling_size': max_pooling_size,
                                                        'convolutional_activation': convolutional_activation,
                                                        'convolutional': convolutional,
                                                        'kernel_try': kernel_try,
                                                        'kernel_size': kernel_size,
                                                        'dropout': dropout,
                                                        'dense_activation': dense_activation,
                                                        'node_variation': node_variation,
                                                        'node_num': node_num,
                                                        'layer': layer,
                                                        'batch_size': batch_size,
                                                        'epoch': epoch
                                                    }
                                                    parameters_list.append(parameters_dict)
                                                    if not combine and (not first_convolutional_activation
                                                                    or not first_convolutional 
                                                                    or not first_kernel_try 
                                                                    or not first_kernel_size 
                                                                    or not first_dropout 
                                                                    or not first_dense_activation 
                                                                    or not first_node_variation
                                                                    or not first_node_num
                                                                    or not first_layer
                                                                    or not first_batch_size
                                                                    or not first_epoch ):
                                                        break
                                                if not combine and (not first_convolutional
                                                                    or not first_kernel_try 
                                                                    or not first_kernel_size 
                                                                    or not first_dropout 
                                                                    or not first_dense_activation 
                                                                    or not first_node_variation
                                                                    or not first_node_num
                                                                    or not first_layer
                                                                    or not first_batch_size
                                                                    or not first_epoch ):
                                                    break
                                            if not combine and (not first_kernel_try 
                                                                or not first_kernel_size 
                                                                or not first_dropout 
                                                                or not first_dense_activation 
                                                                or not first_node_variation
                                                                or not first_node_num
                                                                or not first_layer
                                                                or not first_batch_size
                                                                or not first_epoch ):
                                                break
                                        if not combine and (not first_kernel_size 
                                                            or not first_dropout 
                                                            or not first_dense_activation 
                                                            or not first_node_variation
                                                            or not first_node_num
                                                            or not first_layer
                                                            or not first_batch_size
                                                            or not first_epoch ):
                                            break
                                    if not combine and (not first_dropout 
                                                        or not first_dense_activation 
                                                        or not first_node_variation
                                                        or not first_node_num
                                                        or not first_layer
                                                        or not first_batch_size
                                                        or not first_epoch ):
                                        break
                                if not combine and (not first_dense_activation 
                                                    or not first_node_variation
                                                    or not first_node_num
                                                    or not first_layer
                                                    or not first_batch_size
                                                    or not first_epoch ):
                                    break
                            if not combine and ( not first_node_variation 
                                                or not first_node_num
                                                or not first_layer
                                                or not first_batch_size
                                                or not first_epoch ):
                                break
                        if not combine and (not first_node_num
                                            or not first_layer
                                            or not first_batch_size
                                            or not first_epoch):
                            break
                    if not combine and (not first_layer
                                        or not first_batch_size
                                        or not first_epoch ):
                        break
                if not combine and (not first_batch_size
                                    or not first_epoch ):
                    break
            if not combine and not first_epoch:
                break

    return parameters_list

def get_model(parameters):
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    print(parameters)

    # Declare an empty list of layers to add to the model
    layers = []

    # Define the convolutional parameters and amount of layers
    for i in range(parameters['convolutional']):
        convolutional_layer = tf.keras.layers.Conv2D( filters=parameters['kernel_try'], 
                                        kernel_size=parameters['kernel_size'],
                                        activation= parameters['convolutional_activation'],
                                        input_shape = (IMG_WIDTH, IMG_HEIGHT, 3))
        layers.append(convolutional_layer)
        pooling_layer = tf.keras.layers.MaxPooling2D((parameters['max_pooling_size'],parameters['max_pooling_size']))
        layers.append(pooling_layer)

    layers.append(tf.keras.layers.Flatten())

    
    layers_num = parameters['layer']

    # We add the hidden layers with the specific nodes amount.
    for i in range(layers_num):
        nodes = parameters['node_num']
        mean_nodes = round( nodes/layers_num, 0 )
        if parameters['node_variation'] == 'equal':
            nodes = mean_nodes
        elif parameters['node_variation'] == 'ascending':
            nodes = round( ( mean_nodes * i / ( layers_num - 1 ) ) + ( mean_nodes / 2 ), 0 )
        elif parameters['node_variation'] == 'descending':
            nodes = round( ( - mean_nodes * i / ( layers_num - 1 ) ) + ( mean_nodes / 2 + mean_nodes ), 0 )
        elif parameters['node_variation'] == 'interleaved':
            nodes = round( mean_nodes * 0.5 * ( i%2 * 2 - 1 ) + mean_nodes, 0 )
        print(f"layer: {i}, nodes: {nodes}")
        dense_layer = tf.keras.layers.Dense(nodes, activation=parameters['dense_activation'])
        layers.append(dense_layer)
    
    # We add one dropout layer
    dropout_layer = tf.keras.layers.Dropout(parameters['dropout'])
    layers.append(dropout_layer)

    # Finally we add the output layer
    output_layer = tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    layers.append(output_layer)


    model = tf.keras.models.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())
    return model



def test_all_parameters(images, labels, comb, splits = 1):


    dict_list = []
    for i in range(splits):
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(images), np.array(labels), test_size=TEST_SIZE
        )
        for ii, parameters in enumerate(comb):
            print()
            print('ITERATION NUMBER')
            print(f"{(ii+1)*(i+1)}/{len(comb)*(splits)}")
            print()
            model = get_model(parameters)
            start = time.time()
            model.fit(x_train, y_train, epochs=parameters['epoch'], batch_size=parameters['batch_size'])
            end = time.time()

            result = model.evaluate(x_test, y_test, verbose=1, return_dict=True)
            result['time'] = end-start
            result['efficiency'] = result['accuracy'] / ( X * result['time'] )
            result = {**result, **parameters}
            result['split_iteration'] = i
            dict_list.append(result)
            print()
            print('RESULT')
            print(result)
            print()

    result_DataFrame = pd.DataFrame(dict_list)
    return result_DataFrame





def main():

    images, labels = load_data(images_main_folder)
    labels = tf.keras.utils.to_categorical(labels)

    comb = create_parameters_combination(   combine = True, 
                                            epochs = [20], # Choose the lowest with min 90 accuracy -> 20
                                            batch_sizes = [60], # Choose the highest -> 60
                                            layers = [6,8,10,12], # Iterate around 10
                                            nodes = [1800,2000,2200], # Iterate around 2000
                                            node_variations = ['equal'],  # Choose equall sinde it does not affect
                                            dense_activations = ['relu','gelu','tanh','sigmoid','exponential','linear'],
                                            dropouts = [0.3], 
                                            kernel_sizes = [3], 
                                            kernel_tries = [32,64], 
                                            convolutionals = [1,2],
                                            convolutional_activations = ['relu'],
                                            max_pooling_sizes = [2] )

    result_dataframe = test_all_parameters(images, labels, comb, splits=2)

    result_dataframe.to_csv('analytics.csv')


if __name__ == '__main__':
    main()