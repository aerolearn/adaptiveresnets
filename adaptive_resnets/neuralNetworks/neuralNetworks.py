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

# Custom layers

class ConvexCombination(tf.keras.layers.Layer):
	"""
	
	"""
	def __init__(self):
		super(ConvexCombination, self).__init__()

	def build(self, input_shape):
		self.w = self.add_weight('convex_combo',shape=(1),
							initializer='zeros',trainable=True)
		
	def call(self, inputs):
		assert len(inputs) == 2
		convex_combo_coefficient = tf.keras.activations.sigmoid(self.w)
		return (1. - convex_combo_coefficient)*inputs[0] +\
						convex_combo_coefficient*inputs[1]
	
class LinearCombination(tf.keras.layers.Layer):
	"""

	"""
	def __init__(self, n_sum = 2):
		super(PositiveLinearCombination, self).__init__()
		self.n_sum = n_sum

	def build(self, input_shape):
		self.w = self.add_weight('linear_combo',shape=(self.n_sum),
							initializer='zeros',trainable=True)

	def call(self, inputs):
		assert len(inputs) == self.n_sum
		coefficient = tf.keras.activations.sigmoid(self.w)
		my_sum = 0.0
		for i in range(self.n_sum):
			my_sum += self.w[i]*inputs[i]
		return my_sum

def low_rank_layer(input_x,rank = 8,activation = 'softplus',name_prefix = None,zeros = True):
	"""
	"""
	output_shape = input_x.shape
	assert len(output_shape) == 2
	output_dim = output_shape[-1]
	if name_prefix is None:
		if zeros:
			intermediate = tf.keras.layers.Dense(rank,activation = activation)(input_x)
			return tf.keras.layers.Dense(output_dim,
										kernel_initializer = tf.keras.initializers.Zeros(),
										bias_initializer = tf.keras.initializers.Zeros())(intermediate)
		else:
			intermediate = tf.keras.layers.Dense(rank,activation = activation)(input_x)
			return tf.keras.layers.Dense(output_dim)(intermediate)
	else:
		if zeros:
			intermediate = tf.keras.layers.Dense(rank,activation = activation,name = name_prefix+'low_rank_in')(input_x)
			return tf.keras.layers.Dense(output_dim,name = name_prefix+'low_rank_out',
										kernel_initializer = tf.keras.initializers.Zeros(),
										bias_initializer = tf.keras.initializers.Zeros())(intermediate)
		else:
			intermediate = tf.keras.layers.Dense(rank,activation = activation,name = name_prefix+'low_rank_in')(input_x)
			return tf.keras.layers.Dense(output_dim,name = name_prefix+'low_rank_out')(intermediate)

# Custom networks for one input and one output

def fixed_output_network(input_dimension,last_layer_weights,hidden_layer_dimensions = [],\
							trainable = False,name_prefix = ''):
	"""
	This network learns coefficients in a pre-determined reduced basis for the output

	name_prefix is to avoid clashing name spaces when we use the model to create multi fidelity surrogates
	"""
	assert type(last_layer_weights) is list
	assert len(last_layer_weights) == 2
	reduced_output_dim, output_dim = last_layer_weights[0].shape
	# Check shape interface conditions
	assert len(last_layer_weights[1].shape) == 1
	assert last_layer_weights[1].shape[0] == output_dim
	assert hidden_layer_dimensions[-1] == reduced_output_dim
	# Define the input layer
	input_data = tf.keras.layers.Input(shape=(input_dimension,),name = name_prefix+'input_data')
	z = input_data
	for i,hidden_layer_dimension in enumerate(hidden_layer_dimensions):
		# Add Glorot initialization here
		z = tf.keras.layers.Dense(hidden_layer_dimension,activation = 'softplus',name = name_prefix+'hidden_layer_'+str(i),\
											kernel_initializer = tf.keras.initializers.GlorotNormal(),
											bias_initializer = tf.keras.initializers.GlorotNormal())(z)
		# z = tf.keras.layers.LeakyReLU()(z)

	output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix+'output_layer')(z)

	network = tf.keras.models.Model(input_data,output_layer)

	network.get_layer(name_prefix+'output_layer').trainable =  trainable
	network.get_layer(name_prefix+'output_layer').set_weights(last_layer_weights)

	return network

def fixed_output_resnet(input_dimension,last_layer_weights,ranks = [],\
							trainable = False,name_prefix = ''):
	"""
	This network learns coefficients in a pre-determined reduced basis for the output

	Uses low rank residual layers

	All layers must have specific names for weight setting in the adaptive construction strategy
	"""
	assert type(last_layer_weights) is list
	assert len(last_layer_weights) == 2
	reduced_output_dim, output_dim = last_layer_weights[0].shape
	# Check shape interface conditions
	assert len(last_layer_weights[1].shape) == 1
	assert last_layer_weights[1].shape[0] == output_dim
	# assert hidden_layer_dimensions[-1] == reduced_output_dim

	# What are the conditions for the input layer?
	# Define the input layer
	input_data = tf.keras.layers.Input(shape=(input_dimension,),name = name_prefix+'input_data')
	z = input_data
	if not input_dimension == reduced_output_dim:
		z = tf.keras.layers.Dense(reduced_output_dim,name = name_prefix+'input_resnet_interface',use_bias=False)(z)
		# z.trainable = False

	for i,rank in enumerate(ranks):
		# Add Glorot initialization here?
		z = tf.keras.layers.Add(name = name_prefix+'add'+str(i))([low_rank_layer(z,rank = rank,activation = 'softplus',name_prefix=name_prefix+str(i)),z])

	output_layer = tf.keras.layers.Dense(output_dim,name = name_prefix+'output_layer')(z)

	network = tf.keras.models.Model(input_data,output_layer)

	network.get_layer(name_prefix+'output_layer').trainable =  trainable
	network.get_layer(name_prefix+'output_layer').set_weights(last_layer_weights)

	return network


def generic_dense(input_dim,output_dim,n_hidden_neurons):
	"""
	"""
	assert type(n_hidden_neurons) is list

	input_data = tf.keras.layers.Input(shape=(input_dim,))
	z = input_data
	for n_hidden_neuron in n_hidden_neurons:	
		z = tf.keras.layers.Dense(n_hidden_neuron, activation='softplus')(z)
	output = tf.keras.layers.Dense(output_dim)(z)
	regressor = tf.keras.models.Model(input_data, output)
	return regressor


