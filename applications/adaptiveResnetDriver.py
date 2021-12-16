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



# Parse run specifications
from argparse import ArgumentParser

parser = ArgumentParser(add_help=True)
parser.add_argument('-data_file', dest='data_file',required=False, default = 'data/dataSets_L3_480.pickle',\
                                                                                         help='location for data',type=str)
parser.add_argument('-OMP_NUM_THREADS', dest='OMP_NUM_THREADS',required=False, default = '4',\
                                                                        help='env variable for openmp parallelism',type=str)
parser.add_argument('-data_repetitions',dest = 'data_repetitions',required= False,default = 20,help='number of data seeds used',type = int)
parser.add_argument('-breadth_tolerance',dest = 'breadth_tolerance',required= False,default = 5e2,\
                                                                                help='relative tolerance for POD',type = int)
parser.add_argument('-max_breadth',dest = 'max_breadth',required= False,default = 10,\
                                                                                help='maximum POD rank',type = int)
parser.add_argument('-depth_start',dest = 'depth_start',required= False,default = 2,\
                                                                        help='depth start for adaptive resnet construction',type = int)
parser.add_argument('-depth_end',dest = 'depth_end',required= False,default = 5,\
                                                                        help='depth end for adaptive resnet construction',type = int)
parser.add_argument('-layer_rank',dest = 'layer_rank',required= False,default = 4,\
                                                                        help='layer rank for adaptive resnets',type = int)
parser.add_argument('-keras_epochs',dest = 'keras_epochs',required= False,default = 150,\
                                                            help='number of epochs for keras training used for non adaptive trainings',type = int)
# Booleans for what networks to traing
parser.add_argument('-train_adaptive_resnet',dest = 'train_adaptive_resnet',required= False,default = 1,\
                                                                        help='integer boolean for training adaptive resnet',type = int)
parser.add_argument('-train_resnet_oneshot',dest = 'train_resnet_oneshot',required= False,default = 1,\
                                                                        help='integer boolean for training resnet one shot',type = int)
parser.add_argument('-train_pod_network',dest = 'train_pod_network',required= False,default = 0,\
                                                                        help='integer boolean for training pod network',type = int)
parser.add_argument('-train_broad_dense',dest = 'train_broad_dense',required= False,default = 1,\
                                                                        help='integer boolean for training broad dense',type = int)
parser.add_argument('-train_medium_dense',dest = 'train_medium_dense',required= False,default = 1,\
                                                                        help='integer boolean for training medium dense',type = int)
parser.add_argument('-train_trim_dense',dest = 'train_trim_dense',required= False,default = 0,\
                                                                        help='integer boolean for training trim dense',type = int)
parser.add_argument('-train_twist_network',dest = 'train_twist_network',required= False,default = 1,\
                                                                        help='integer boolean for training twist (3d only)',type = int)
parser.add_argument('-twist_depth',dest = 'twist_depth',required= False,default = 2,\
                                                                        help='depth for twist network',type = int)
args = parser.parse_args()

import os, sys
import pickle,time, datetime
import numpy as np

# Tensorflow imports and default setting
# Before importing must set the following
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["KMP_WARNINGS"] = "FALSE" 
os.environ['OMP_NUM_THREADS'] = args.OMP_NUM_THREADS

import tensorflow as tf
if int(tf.__version__[0]) > 1:
    import tensorflow.compat.v1 as tf
    # tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)
# Tensorflow seed and data type settings
try:
    tf.random.set_seed(42)
except:
    tf.set_random_seed(42)
tf.keras.backend.set_floatx('float64')
# Local imports
sys.path.append('../')
from adaptiveresnets import *

logger_name = 'master_logger_multirun'
logger_name += str(datetime.date.today())
# Allocate each individual logger
twist_generic_logger = {}
adaptive_resnet_logger = {}
one_shot_resnet_logger = {}
pod_network_logger = {}
broad_generic_logger = {}
medium_generic_logger = {}
trim_generic_logger = {}

# Load the data here
f = open(args.data_file, 'rb')
masterdict = pickle.load(f)
f.close()

if args.train_twist_network:
    for key in masterdict:
        assert 't_train' in masterdict[key].keys()

for key in list(masterdict.keys())[:args.data_repetitions]:
    # Instance nested loggers for this rep seed for each network
    twist_generic_logger[key] = {}
    adaptive_resnet_logger[key] = {}
    one_shot_resnet_logger[key] = {}
    pod_network_logger[key] = {}
    broad_generic_logger[key] = {}
    medium_generic_logger[key] = {}
    trim_generic_logger[key] = {}


    print('Running neural networks for key ',key)
    mydict = masterdict[key]

    # Check how much training and validation data are available
    n_train_total = mydict['d_train'].shape[0]
    n_val_total = mydict['d_valid'].shape[0]
    print(80*'#')
    print('Number of training data available: ',n_train_total)
    print('Number of validation data available: ',n_val_total)
    print(80*'#')
    
    for portion in [0.05,0.1,0.2,0.3,0.4,0.5,0.75,1.0]:
        print('Running for key = ',key, 'out of  ', args.data_repetitions)
        n_train = int(portion*n_train_total)
        n_val = int(portion*n_val_total)
        print('n_train, n_val',(n_train,n_val))


        # Parse the data and instance the data dictionary
        d_train = mydict['d_train'][:n_train]
        x_train = mydict['x_train'][:n_train]
        d_valid = mydict['d_valid'][:n_val]
        x_valid = mydict['x_valid'][:n_val]
        d_test  = mydict['d_test']
        x_test = mydict['x_test']

        data_dict = {'input_train':d_train,'output_train':x_train,
                    'input_val':d_valid, 'output_val':x_valid,
                    'input_test':d_test, 'output_test':x_test}
        # First step adaptive resnet training
        parameters = adaptive_resnet_parameters()
        parameters['breadth_tolerance'] = args.breadth_tolerance
        parameters['max_breadth'] = args.max_breadth
        parameters['depth_start'] = args.depth_start
        parameters['depth_end'] = args.depth_end
        # Actually do this manualy based on epochs / number of adaptive layers
        adaptive_epochs = args.keras_epochs /(args.depth_end - args.depth_start)
        print('adaptive_epochs = ',adaptive_epochs)
        parameters['keras_epochs'] = adaptive_epochs
        parameters['layer_rank'] = args.layer_rank

        if args.train_adaptive_resnet:
            # Train and evaluate resnets adaptively
            trained_resnet, adaptive_logger = adaptive_resnet_construction_fixed_depths(data_dict,parameters = parameters)
            adaptive_resnet_logger[key][n_train] = adaptive_logger
        else:
            adaptive_resnet_logger[key][n_train] = {}

        parameters['keras_epochs'] = args.keras_epochs

        # Compute parameters used for the construction of comparable baseline networks
        depth = parameters['depth_end']
        input_dim = data_dict['input_train'].shape[-1]
        output_dim = data_dict['output_train'].shape[-1]
        n_train = data_dict['input_train'].shape[0]

        if args.train_resnet_oneshot:
            # Fixed output POD resnet trained in one shot
            resnet_ranks = depth*[parameters['layer_rank']]
            pod_resnet  = construct_pod_resnet(data_dict,ranks = resnet_ranks,parameters = parameters)
            print(80*'#')
            print('Training POD resnet in one shot'.center(80))
            one_shot_trained_resnet, one_shot_resnet_logger_i = one_shot_training(pod_resnet, data_dict)
            one_shot_resnet_logger[key][n_train] = one_shot_resnet_logger_i
        else:
            one_shot_resnet_logger[key][n_train] = {}

        if args.train_pod_network:
            # Fixed output POD dense network
            pod_network = construct_pod_dense(data_dict,depth = depth,\
                                               breadth_tolerance = parameters['breadth_tolerance'],\
                                                max_breadth = parameters['max_breadth'])
            print(80*'#')
            print('Training POD dense network'.center(80))
            pod_network, pod_network_logger_i = one_shot_training(pod_network, data_dict)
            pod_network_logger[key][n_train] = pod_network_logger_i
        else:
            pod_network_logger[key][n_train] = {}

        if args.train_broad_dense:
            # Broad generic dense network
            broad_generic_network = generic_dense(input_dim,output_dim,depth*[output_dim])
            print(80*'#')
            print('Training broad generic dense network'.center(80))
            broad_generic_network, broad_generic_logger_i = one_shot_training(broad_generic_network, data_dict)
            broad_generic_logger[key][n_train] = broad_generic_logger_i
        else:
            broad_generic_logger[key][n_train] = {}

        if args.train_medium_dense:
            # Medium generic dense network that uses the same intermediate size as the resnet / POD networks
            last_layer_weights = build_POD_layer_arrays(data_dict,\
                                        breadth_tolerance = parameters['breadth_tolerance'],\
                                            max_breadth = parameters['max_breadth']) 
            truncation_dimension = last_layer_weights[0].shape[0]
            medium_generic_network = generic_dense(input_dim,output_dim,depth*[truncation_dimension])
            print(80*'#')
            print('Training medium generic dense network'.center(80))
            medium_generic_network, medium_generic_logger_i = one_shot_training(medium_generic_network, data_dict)
            medium_generic_logger[key][n_train] = medium_generic_logger_i
        else:
            medium_generic_logger[key][n_train] = {}

        if args.train_trim_dense:
            # Trim generic dense network
            trim_generic_network = generic_dense(input_dim,output_dim,depth*[input_dim])
            print(80*'#')
            print('Training trim generic dense network'.center(80))
            trim_generic_network, trim_generic_logger_i = one_shot_training(trim_generic_network, data_dict)
            trim_generic_logger[key][n_train] = trim_generic_logger_i
        else:
            trim_generic_logger[key][n_train] = {}

        if args.train_twist_network:
            print(80*'#')
            print('Twist training'.center(80))
            t_train = mydict['t_train'][:n_train]
            t_valid = mydict['t_valid'][:n_val]
            t_test = mydict['t_test']
            twist_data_dict = {'input_train':d_train,'output_train':t_train,
                                'input_val':d_valid, 'output_val':t_valid,
                                'input_test':d_test, 'output_test':t_test}
            twist_output_dim = twist_data_dict['output_train'].shape[-1]
            # Trim generic dense network
            twist_last_layer_weights = build_POD_layer_arrays(twist_data_dict,\
                                        breadth_tolerance = parameters['breadth_tolerance'],\
                                            max_breadth = parameters['max_breadth']) 
            twist_truncation_dimension = twist_last_layer_weights[0].shape[0]
            twist_network = generic_dense(input_dim,twist_output_dim,args.twist_depth*[input_dim])
            twist_network, twist_generic_logger_i = one_shot_training(twist_network, twist_data_dict)
            twist_generic_logger[key][n_train] = twist_generic_logger_i
        else:
            twist_generic_logger[key][n_train] = {}

    # Important: save for each seed, since these things take a while
    # If this becomes parallelized then maybe seeds need to be pulled out
    # and saved independently to avoid concurrent writes
    # Post-processing to save a big pkl file
    master_logger = {}
    master_logger['adaptive_resnet'] = adaptive_resnet_logger
    master_logger['one_shot_resnet'] = one_shot_resnet_logger
    master_logger['pod_network'] = pod_network_logger
    master_logger['broad_generic_network'] = broad_generic_logger
    master_logger['medium_generic_network'] = medium_generic_logger
    master_logger['trim_generic_network'] = trim_generic_logger
    master_logger['twist_generic_network'] = twist_generic_logger

    try:
        os.makedirs('logging/')
    except:
        pass

    if not logger_name.endswith('.pkl'):
        logger_name += '.pkl'
    with open('logging/'+logger_name, 'wb+') as f:
        pickle.dump(master_logger, f, pickle.HIGHEST_PROTOCOL)


