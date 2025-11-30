"""
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import numpy as np
import tensorflow as tf

def is_FC(lname):
	"""
	"""
	import re
	pattns = ['Dense*'] # now, only Dense
	return any([bool(re.match(t,lname)) for t in pattns])

def is_C2D(lname):
	"""
	"""
	import re
	pattns = ['Conv2D']
	return any([bool(re.match(t,lname)) for t in pattns])

def is_LSTM(lname):
	"""
	"""
	import re
	pattns = ['.*LSTM*']
	return any([bool(re.match(t,lname)) for t in pattns])

def is_BatchNorm(lname):
	"""
	"""
	import re
	pattns = ['.*[Bb]atch.*[Nn]orm']
	return any([bool(re.match(t,lname)) for t in pattns])

def is_Input(lname):
	"""
	"""
	import re
	pattns = ['InputLayer']
	return any([bool(re.match(t,lname)) for t in pattns])


def convert_cifar_to_channels_last(src_path, dst_path):
	"""
	Convert a channels_first CIFAR10 model (simple_cm) to channels_last for CPU usage.
	Assumes architecture: Input -> ZeroPadding2D -> Conv2D -> BatchNorm -> Activation ->
	MaxPooling2D -> Reshape -> Dense -> Activation -> Dense.
	"""
	if os.path.exists(dst_path):
		return dst_path
	from tensorflow.keras import layers, models
	src = tf.keras.models.load_model(src_path, compile=False)
	assert len(src.layers) >= 10, "Unexpected model architecture"

	l_pad = src.layers[1]
	l_conv = src.layers[2]
	l_bn = src.layers[3]
	l_act1 = src.layers[4]
	l_pool = src.layers[5]
	l_dense1 = src.layers[7]
	l_act2 = src.layers[8]
	l_dense2 = src.layers[9]

	inp = layers.Input(shape=(32,32,3))
	x = layers.ZeroPadding2D(padding=l_pad.padding, data_format='channels_last')(inp)
	x = layers.Conv2D(
		filters=l_conv.filters,
		kernel_size=l_conv.kernel_size,
		strides=l_conv.strides,
		padding=l_conv.padding,
		use_bias=l_conv.use_bias,
		data_format='channels_last')(x)
	x = layers.BatchNormalization(
		axis=-1,
		epsilon=l_bn.epsilon,
		momentum=l_bn.momentum)(x)
	x = layers.Activation(l_act1.activation)(x)
	x = layers.MaxPooling2D(
		pool_size=l_pool.pool_size,
		strides=l_pool.strides,
		padding=l_pool.padding,
		data_format='channels_last')(x)
	x = layers.Reshape((np.prod(x.shape[1:]),))(x)
	x = layers.Dense(l_dense1.units, use_bias=True)(x)
	x = layers.Activation(l_act2.activation)(x)
	out = layers.Dense(l_dense2.units, activation=l_dense2.activation)(x)

	dst_model = models.Model(inp, out)
	# transfer weights
	dst_model.layers[2].set_weights(l_conv.get_weights())
	dst_model.layers[3].set_weights(l_bn.get_weights())
	dst_model.layers[7].set_weights(l_dense1.get_weights())
	dst_model.layers[9].set_weights(l_dense2.get_weights())

	os.makedirs(os.path.dirname(dst_path), exist_ok=True)
	dst_model.save(dst_path)
	return dst_path


def convert_gtsrb_to_channels_last(src_path, dst_path):
	"""
	Convert channels_first GTSRB model to channels_last.
	Architecture: Input -> Conv2D -> BatchNorm -> MaxPool -> Flatten -> Dense -> BatchNorm -> Dense.
	"""
	if os.path.exists(dst_path):
		return dst_path
	from tensorflow.keras import layers, models
	src = tf.keras.models.load_model(src_path, compile=False)
	assert len(src.layers) >= 8, "Unexpected GTSRB model architecture"

	l_conv = src.layers[1]
	l_bn1 = src.layers[2]
	l_pool = src.layers[3]
	l_dense1 = src.layers[5]
	l_bn2 = src.layers[6]
	l_dense2 = src.layers[7]

	inp = layers.Input(shape=(48,48,3))
	x = layers.Conv2D(
		filters=l_conv.filters,
		kernel_size=l_conv.kernel_size,
		strides=l_conv.strides,
		padding=l_conv.padding,
		use_bias=l_conv.use_bias,
		data_format='channels_last')(inp)
	x = layers.BatchNormalization(
		axis=-1,
		epsilon=l_bn1.epsilon,
		momentum=l_bn1.momentum)(x)
	x = layers.MaxPooling2D(
		pool_size=l_pool.pool_size,
		strides=l_pool.strides,
		padding=l_pool.padding,
		data_format='channels_last')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(l_dense1.units, use_bias=True)(x)
	x = layers.BatchNormalization(
		axis=-1,
		epsilon=l_bn2.epsilon,
		momentum=l_bn2.momentum)(x)
	out = layers.Dense(l_dense2.units, activation=l_dense2.activation)(x)

	dst_model = models.Model(inp, out)
	# transfer weights (conv/bn/dense/bn/dense)
	dst_model.layers[1].set_weights(l_conv.get_weights())
	dst_model.layers[2].set_weights(l_bn1.get_weights())
	dst_model.layers[5].set_weights(l_dense1.get_weights())
	dst_model.layers[6].set_weights(l_bn2.get_weights())
	dst_model.layers[7].set_weights(l_dense2.get_weights())

	os.makedirs(os.path.dirname(dst_path), exist_ok=True)
	dst_model.save(dst_path)
	return dst_path


def get_loss_func(is_multi_label = True):
	"""
	here, we will only return either cross_entropy or binary_crossentropy
	"""
	loss_func = 'categorical_cross_entropy' if is_multi_label else 'binary_crossentropy'
	return loss_func 


def predict_with_new_delat(
	fn_mdl, deltas, min_idx_to_tl, 
	init_biases, init_weights, 
	prev_outputs, chunks):
	"""
	predict with the model patched using deltas
	"""
	from collections.abc import Iterable
	for idx_to_tl, delta in deltas.items(): 
		# either idx_to_tl or (idx_to_tl, i)
		if isinstance(idx_to_tl, Iterable):
			idx_to_t_mdl_l, idx_to_w = idx_to_tl
		else:
			idx_to_t_mdl_l = idx_to_tl
	
		# index of idx_to_tl (from deltas) in the local model
		local_idx_to_l = idx_to_t_mdl_l - min_idx_to_tl + 1 
		lname = type(fn_mdl.layers[local_idx_to_l]).__name__
		if is_FC(lname) or is_C2D(lname):
			fn_mdl.layers[local_idx_to_l].set_weights([delta, init_biases[idx_to_t_mdl_l]])
		elif is_LSTM(lname):
			if idx_to_w == 0: # kernel
				new_kernel_w = delta # use the full 
				new_recurr_kernel_w = init_weights[(idx_to_t_mdl_l, 1)]
			elif idx_to_w == 1:
				new_recurr_kernel_w = delta
				new_kernel_w = init_weights[(idx_to_t_mdl_l, 0)]
			else:
				print ("{} not allowed".format(idx_to_w), idx_to_t_mdl_l, idx_to_tl)
				assert False
			# set kernel, recurr kernel, bias
			fn_mdl.layers[local_idx_to_l].set_weights(
				[new_kernel_w, new_recurr_kernel_w, init_biases[idx_to_t_mdl_l]])
		else:
			print ("{} not supported".format(lname))
			assert False

	predictions = None
	for chunk in chunks:
		_predictions = fn_mdl.predict(prev_outputs[chunk], batch_size = len(chunk))
		if predictions is None:
			predictions = _predictions
		else:
			predictions = np.append(predictions, _predictions, axis = 0)

	return predictions
