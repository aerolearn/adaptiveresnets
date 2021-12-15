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

from .neuralNetworks import *


# Define procedures for defining networks via POD spectral decay here

def build_POD_layer_arrays(data_dict,breadth_tolerance = 1e2, max_breadth = 10):
	"""
	"""
	U, s, _ = np.linalg.svd((data_dict['output_train']-np.mean(data_dict['output_train'],axis=0)).T,\
							full_matrices = False)
	orders_reduction = np.array([s[0]/(si+1e-12) for si in s])
	# Absolute tolerance for breadth
	abs_tol_idx = np.where(orders_reduction>breadth_tolerance)[0][0]
#     print('abs tol idx = ',abs_tol_idx)
	truncation_dimension = min(abs_tol_idx,max_breadth)
	
	U = U[:,:truncation_dimension]
	last_layer_weights = [U.T,np.mean(data_dict['output_train'], axis=0)]

	return last_layer_weights


def construct_pod_resnet(data_dict, ranks, parameters, name_prefix = ''):
	"""
	"""
	last_layer_weights = build_POD_layer_arrays(data_dict,breadth_tolerance = parameters['breadth_tolerance'],\
																		max_breadth = parameters['max_breadth'])
	input_dimension = data_dict['input_train'].shape[-1]
	pod_resnet = fixed_output_resnet(input_dimension,last_layer_weights,\
									ranks = ranks,name_prefix = name_prefix)

	return pod_resnet

def construct_pod_dense(data_dict, depth, breadth_tolerance = 1e2, max_breadth = 10, name_prefix = ''):
	"""
	"""
	last_layer_weights = build_POD_layer_arrays(data_dict,breadth_tolerance = breadth_tolerance,\
																		max_breadth = max_breadth)
	truncation_dimension = last_layer_weights[0].shape[0]
	input_dimension = data_dict['input_train'].shape[-1]
	pod_dense_network = fixed_output_network(input_dimension,last_layer_weights,\
									hidden_layer_dimensions = depth*[truncation_dimension],name_prefix = name_prefix)
	return pod_dense_network

