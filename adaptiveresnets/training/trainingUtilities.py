# Copyright (c) 2021, The University of Texas at Austin 
# & University of Michigan
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the aerolearn package. For more information see
# https://github.com/aerolearn/aerolearn/
#
# aerolearn is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import tensorflow as tf
import numpy as np

# Defining accuracy and metrics here for use

def root_mean_squared_error(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

def l2_accuracy(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    normalized_squared_difference = tf.reduce_mean(squared_difference,axis=-1)\
                                    /tf.reduce_mean(tf.square(y_true),axis =-1)
    return 1. - tf.sqrt(tf.reduce_mean(normalized_squared_difference))

def l1_accuracy(y_true, y_pred):
    abs_difference = tf.abs(y_true - y_pred)
    normalized_abs_difference = tf.reduce_mean(abs_difference,axis = -1)\
                        /tf.reduce_mean(tf.abs(y_true),axis = -1)
    return 1. - tf.reduce_mean(normalized_abs_difference)


# Network training parameters

def network_training_parameters():
    parameters = {}
    parameters['keras_epochs'] = 1000
    parameters['batch_size'] = 2
    parameters['optimizer'] = 'adam' # choose from adam / SGD / whatever keras optimizers
    parameters['step_length'] = 0.001

    return parameters

# Network training callback
def train_network(network,data_dict,opt_parameters = network_training_parameters(),verbose = False):
    settings = {}
    # Callbacks for training
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=opt_parameters['keras_epochs'])
    checkpoint_filepath = '/tmp/checkpoint'+'L1SF'
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss',\
                                            mode='min', verbose=0, save_best_only=True,save_weights_only=True)
    callbacks = [early_stop,model_checkpoint]
    if opt_parameters['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = opt_parameters['step_length'])
    elif opt_parameters['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = opt_parameters['step_length'])
    else:
        raise 'Invalid choice of optimizer'

    network.compile(optimizer=optimizer,loss=root_mean_squared_error,metrics=[l1_accuracy,l2_accuracy])
    
    if verbose:
        network.evaluate(data_dict['input_test'],data_dict['output_test'])

    network.fit(data_dict['input_train'],data_dict['output_train'],
                validation_data = (data_dict['input_val'],data_dict['output_val']),epochs = opt_parameters['keras_epochs'],\
                                    batch_size = opt_parameters['batch_size'],verbose = False,callbacks = callbacks)
#     if keras_epochs == 0:
#         network.load_weights(checkpoint_filepath)
    if verbose:
        print('After keras training')
        network.evaluate(data_dict['input_test'],data_dict['output_test'])

# Train the networks in one go

def one_shot_training(network, data_dict,  opt_parameters = network_training_parameters()):
    logger = {}
    trainable_count = np.sum([np.prod(v.get_shape()) for v in network.trainable_weights])
    print('Training network in one shot, dW = ',trainable_count)
    train_network(network,data_dict,opt_parameters = opt_parameters,verbose = False)
    evaluated_metrics = network.evaluate(data_dict['input_test'],data_dict['output_test'],verbose = False) 
    print('After one shot training')
    logger['l1 acc'] = evaluated_metrics[1]
    logger['l2 acc'] = evaluated_metrics[2]
    logger['dW'] = trainable_count
    print('l1 accuracy = ',evaluated_metrics[1])
    print('l2_accuracy = ',evaluated_metrics[2])
    
    return network, logger

