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

import numpy as np

# Adaptive training layer by layer (for ResNets)

def adaptive_resnet_parameters():
    parameters = {}
    parameters['breadth_tolerance'] = 5e2
    parameters['max_breadth'] = 8
    parameters['freeze_at_depth'] = np.inf
    parameters['depth_start'] = 2
    parameters['depth_end'] = 5
    parameters['keras_epochs'] = 1000
    parameters['layer_rank'] = 4
    
    return parameters


def adaptive_resnet_construction_fixed_depths(data_dict, set_weights = {},parameters = adaptive_resnet_parameters(),\
                                                                opt_parameters = network_training_parameters()):
    logger = {}
    evaluated_metrics_old = [100.,0.,0.]
    freeze_at_depth = parameters['freeze_at_depth']
    for depth in range(parameters['depth_start'],parameters['depth_end']+1):
        depth_key = 'depth '+str(depth)
        logger[depth_key] = {}
        set_weights[depth] = {}
        ranks = depth*[parameters['layer_rank']]
        pod_resnet = construct_pod_resnet(data_dict,ranks = ranks,parameters = parameters)
        trainable_count = np.sum([np.prod(v.get_shape()) for v in pod_resnet.trainable_weights])

        print(80*'#')
        print('Constructing resnet for depth',depth, ' dW = ',trainable_count)
        if depth-1 in set_weights.keys():
            for layer_name,weights in set_weights[depth].items():
                pod_resnet.get_layer(layer_name).set_weights(weights)
                if depth >= freeze_at_depth:
                    pod_resnet.get_layer(layer_name).trainable = False
        train_network(pod_resnet,data_dict,opt_parameters = opt_parameters,verbose = False)
        # Post process here and decide if we stop construction
        evaluated_metrics = pod_resnet.evaluate(data_dict['input_val'],data_dict['output_val'],verbose = False)
        # Logging 
        # For a single output network the evaluated metrics should be
        # 0 : loss, 1: metric_1, 2: metric_2 etc.  
        # The convention here is that metric 1 is ell1 acc (taxi-cab norm)
        # and metric 2 is ell2 acc (as the crow flies norm)
        logger[depth_key]['l1 acc'] = evaluated_metrics[1]
        logger[depth_key]['l2 acc'] = evaluated_metrics[2]
        logger[depth_key]['dW'] = trainable_count
        l1_improvement = evaluated_metrics[1]-evaluated_metrics_old[1]
        l2_improvement = evaluated_metrics[2]-evaluated_metrics_old[2]
        # If the network is worse, then freeze and re-train from the old
        # configuration
        if l1_improvement < 0. or l2_improvement < 0.:
            freeze_at_depth = depth
            print('B'+79*'R')
            print('FREEZE TIME'.center(80))
            print('B'+79*'R')
            if depth-1 in set_weights.keys():
                for layer_name,weights in set_weights[depth].items():
                    pod_resnet.get_layer(layer_name).set_weights(weights)
                    pod_resnet.get_layer(layer_name).trainable = False
            train_network(pod_resnet,data_dict,opt_parameters = opt_parameters,verbose = False)
        evaluated_metrics_old = evaluated_metrics
        for layer in pod_resnet.layers:
            set_weights[depth][layer.name] = pod_resnet.get_layer(layer.name).get_weights()
        print('l1  val accuracy = ',evaluated_metrics[1])
        print('l2 val accuracy = ',evaluated_metrics[2])
    # One last training on the outside
    print('Freeing up all weights')
    for layer in pod_resnet.layers:
        if not 'output_layer' in layer.name:
            pod_resnet.get_layer(layer.name).trainable = True

    trainable_count = np.sum([np.prod(v.get_shape()) for v in pod_resnet.trainable_weights])

    print('dW = ',trainable_count)
    train_network(pod_resnet,data_dict,opt_parameters = opt_parameters,verbose = False)
    evaluated_metrics = pod_resnet.evaluate(data_dict['input_test'],data_dict['output_test'],verbose = False) 
    print('After the final training')
    print('l1 test accuracy = ',evaluated_metrics[1])
    print('l2 test accuracy = ',evaluated_metrics[2])
    logger['final network'] = {}
    logger['final network']['l1 acc'] = evaluated_metrics[1]
    logger['final network']['l2 acc'] = evaluated_metrics[2]
    logger['final network']['dW'] = trainable_count

    # Also add going backwards to the specifications of the best previous network
    
    return pod_resnet, logger

