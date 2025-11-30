"""
Localise faults in offline for any faults
"""
import itertools

import numpy as np
import time
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.models import load_model, Model
import tensorflow.compat.v1.keras.backend as K
tf.disable_eager_execution()
from tqdm import tqdm
import lstm_layer
import utils.model_util as model_util

from tensorflow.keras import layers
from tensorflow.keras.layers import CategoryEncoding

from pathlib import Path

# "divided by zeros" is handleded afterward
np.seterr(divide='ignore', invalid='ignore')




class Combine(layers.Layer):
    """Combine outputs of branches based on gate output."""

    def __init__(self, **kwargs):
        """Initializing CombineLayer with the shape of the split."""
        super().__init__()

    def get_config(self):
        """Config information of this layer."""
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs,training=None):
        """Concrete process of combine.

        branch_list = tf.split(input, self.split_shape, 1)
        combined = branch_list[self.inner_predict(branch_list[-1])]
        """
        gate_categories = self.inner_predict(inputs[-1], len(inputs) - 1)
        out_list = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        try:
            for i in range(len(gate_categories)):
                out = inputs[0][i] * gate_categories[i][0]
                for j in range(1, len(inputs) - 1):
                    out = out + inputs[j][i] * gate_categories[i][j]
                out_list = out_list.write(out_list.size(), out)

            final_outs = out_list.stack()

            return final_outs
        except TypeError:
            print(inputs,training,gate_categories,gate_categories.shape)
        #
        #    length = gate_categories.shape[1]
        #    for i in range(length):
        #        out = inputs[0][i] * gate_categories[i][0]
        #        for j in range(1, len(inputs) - 1):
        #            out = out + inputs[j][i] * gate_categories[i][j]
        #        out_list = out_list.write(out_list.size(), out)
        #
        #    final_outs = out_list.stack()
        #    return final_outs

    @tf.function
    def inner_predict(self, data, num):
        """Gating branch based on gating layer."""
        tf_list = tf.math.argmax(data, 1)
        gate_out = CategoryEncoding(num_tokens=num, output_mode="one_hot")(tf_list)
        return gate_out

def load_model_from_h5(model_dir: Path, must_compile=True):
    """
    Loading model from h5
    """

    print("Loading hydra from json and h5")

    json_file = model_dir/Path("model.json")
    h5_file = model_dir / Path("model.h5")

    f = open(json_file,"r")
    description = f.read()
    f.close()

    model = tf.keras.models.model_from_json(description,custom_objects={"Combine":Combine})
    model.load_weights(h5_file)

    if must_compile:
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_target_weights(model, path_to_keras_model, indices_to_target = None):
    """
    return indices to weight layers denoted by indices_to_target, or return all trainable layers

    :param indices_to_target: number of layers to grab as targets. Grab the last X ones that have weights
    """
    import re
    targeting_clname_pattns = ['Dense*', 'Conv*', '.*LSTM*'] #if not target_all else None
    is_target = lambda clname,targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

    if model is None:
        assert path_to_keras_model is not None

        if "hydra" not in str(path_to_keras_model):
            model = load_model(path_to_keras_model, compile=False)
        else:
            model = load_model_from_h5(path_to_keras_model)

    target_weights = {} # key = layer index, value: [weight value, layer name]
    if indices_to_target is not None:
        num_layers = len(model.layers)

        for i in range(num_layers-1, -1, -1):
            layer = model.layers[i]
            ws = layer.get_weights()
            if len(ws) == 0:
                print("the target layer doesn't have weight")
                continue
            elif model_util.is_BatchNorm(type(layer).__name__):
                print(f"skipped layer num {i}, had weights but is of type {type(layer).__name__}")
                continue
            else:
                #target_weights[i] = ws[0] # we will target only the weight, and not the bias
                target_weights[i] = [ws[0], type(layer).__name__]
            if len(target_weights) >= indices_to_target:  # when we found enough layers, break
                print(f"out of {num_layers} layers, we will localize weights in layers: {str(target_weights.keys())}")
                break
    else:
        for i, layer in enumerate(model.layers):
            class_name = type(layer).__name__
            if is_target(class_name, targeting_clname_pattns):
                ws = layer.get_weights()
                if len(ws): # has weight
                    if model_util.is_FC(class_name) or model_util.is_C2D(class_name):
                        target_weights[i] = [ws[0], type(layer).__name__]
                    elif model_util.is_LSTM(class_name):
                        # for LSTM, even without bias, a fault can be in the weights of the kernel
                        # or the recurrent kernel (hidden state handling)
                        assert len(ws) == 3, ws
                        # index 0: for the kernel, index 1: for the recurrent kernel
                        target_weights[i] = [ws[:-1], type(layer).__name__]
                    else:
                        print ("{} not supported yet".format(class_name))
                        assert False

    return target_weights


def _normalise_sample_weights(num, sample_weights):
    """
    Normalise sample weights to sum to 1; returns None when weights are absent or all-zero.
    """
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=float).flatten()
    assert len(weights) == num, f"Weight length {len(weights)} does not match number of samples {num}"
    weights = np.clip(weights, a_min=0.0, a_max=None)
    total = np.sum(weights)
    if total == 0:
        return None
    return weights / total


def _weighted_mean(arr, weights):
    """
    Weighted mean over axis 0; defaults to arithmetic mean when weights are None.
    """
    if weights is None:
        return np.mean(arr, axis=0)
    return np.tensordot(weights, arr, axes=([0], [0]))


def compute_gradient_to_output(path_to_keras_model, 
    idx_to_target_layer, X,
    by_batch = False, on_weight = False, wo_reset = False,federated=False,
    sample_weights = None):
    """
    compute gradients normalisesd and averaged for a given input X
    on_weight = False -> on output of idx_to_target_layer'th layer
    """
    from sklearn.preprocessing import Normalizer
    from collections.abc import Iterable
    norm_scaler = Normalizer(norm = "l1")

    #model = load_model(path_to_keras_model, compile = False)

    if "hydra" not in str(path_to_keras_model):
        model = load_model(path_to_keras_model, compile=False)
    else:
        model = load_model_from_h5(path_to_keras_model)

    if not on_weight:
        target = model.layers[idx_to_target_layer].output
    else: # on weights
        target = model.layers[idx_to_target_layer].weights[:-1]  # exclude the bias

    tensor_grad = tf.gradients(
        model.output,
        target,
        name = 'output_grad')

    # since this might cause OOM error, divide them
    num = X.shape[0]
    if by_batch:
        batch_size = 64
        num_split = int(np.round(num/batch_size))
        if num_split == 0:
            num_split = 1
        chunks = np.array_split(np.arange(num), num_split)
    else:
        chunks = [np.arange(num)]

    weights = _normalise_sample_weights(num, sample_weights)

    if not on_weight:
        grad_shape = tuple([num] + [int(v) for v in tensor_grad[0].shape[1:]])
        gradient = np.zeros(grad_shape)
        for chunk in chunks:
            _gradient = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk]})[0]
            gradient[chunk] = _gradient

        #if not federated:
        #	gradient = np.abs(gradient)
        #else:
        gradient = np.abs(gradient)

        reshaped_gradient = gradient.reshape(gradient.shape[0],-1) # flatten
        norm_gradient = norm_scaler.fit_transform(reshaped_gradient) # normalised


        mean_gradient = _weighted_mean(norm_gradient, weights) # compute mean for a given input

        ret_gradient = mean_gradient.reshape(gradient.shape[1:]) # reshape to the orignal shape
        #print("after mean and reshape", ret_gradient.shape)
        if not wo_reset:
            reset_keras([tensor_grad])
        return ret_gradient
    else: # on a weight variable
        gradients = []
        for chunk in chunks:
            _gradients = K.get_session().run(tensor_grad, feed_dict={model.input: X[chunk]})
            if len(gradients) == 0:
                gradients = _gradients
            else:
                for i in range(len(_gradients)):
                    gradients[i] += _gradients[i]

        ret_gradients = list(map(np.abs, gradients))

        if not wo_reset:
            reset_keras([tensor_grad])

        if len(ret_gradients) == 0:
            return ret_gradients[0]
        else:
            return ret_gradients


def compute_gradient_to_loss(path_to_keras_model, idx_to_target_layer, X, y, 
    by_batch = False, wo_reset = False, loss_func = 'categorical_cross_entropy', federated=False, **kwargs):
    """
    compute gradients for the loss.
    kwargs contains the key-word argumenets required for the loss funation
    """
    #model = load_model(path_to_keras_model, compile = False)

    if "hydra" not in str(path_to_keras_model):
        model = load_model(path_to_keras_model, compile=False)
    else:
        model = load_model_from_h5(path_to_keras_model)

    targets = model.layers[idx_to_target_layer].weights
    if len(targets) > 1:
        targets = targets[:-1]
    output_tensor = model.output
    if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 1:
        output_tensor = tf.squeeze(output_tensor, axis=1)
    if len(output_tensor.shape) == 3:
        y_tensor = tf.keras.Input(shape=(output_tensor.shape[-1],), name='labels')
    else:  # is not multi label
        y_tensor = tf.keras.Input(shape=list(output_tensor.shape)[1:], name='labels')

    if loss_func == 'categorical_cross_entropy':
        # might be changed as the following two
        #loss_tensor = tf.nn.softmax_cross_entropy_with_logits_v2(
        #	labels = y_tensor,
        #	logits = model.output,
        #	name = "per_label_loss")
        loss_tensor = tf.keras.losses.CategoricalCrossentropy()(y_tensor, output_tensor)
    elif loss_func == 'binary_crossentropy':
        if 'name' in kwargs.keys():
            kwargs.pop("name")
        loss_tensor = tf.keras.losses.binary_crossentropy(y_tensor, output_tensor) #y_true, y_pred
        loss_tensor.__dict__.update(kwargs)
        y = y.reshape(-1,1)
    elif loss_func in ['mean_squared_error', 'mse']:
        if 'name' in kwargs.keys():
            kwargs.pop("name")
        loss_tensor = tf.keras.losses.MeanSquaredError(y_tensor, output_tensor, name = "per_label_loss")
        loss_tensor.__dict__.update(kwargs)
    else:
        print (loss_func)
        print ("{} not supported yet".format(loss_func))
        assert False

    tensor_grad = tf.gradients(loss_tensor, targets)
    # since this might cause OOM error, divide them
    num = X.shape[0]
    if by_batch:
        batch_size = 64
        num_split = int(np.round(num/batch_size))
        if num_split == 0:
            num_split += 1
        chunks = np.array_split(np.arange(num), num_split)
    else:
        chunks = [np.arange(num)]

    #gradients = []
    gradients = [[] for _ in range(len(targets))]
    for chunk in chunks:
        _gradients = K.get_session().run(
            tensor_grad, feed_dict={model.input: X[chunk], y_tensor: y[chunk]})
        for i,_gradient in enumerate(_gradients):
            gradients[i].append(_gradient)

    for i, gradients_p_chunk in enumerate(gradients):

        if not federated:
            gradients[i] = np.abs(np.sum(np.asarray(gradients_p_chunk), axis = 0)) # combine
        else:
            gradients[i] = np.sum(np.asarray(gradients_p_chunk), axis=0)  # combine

    if not wo_reset:
        reset_keras(gradients + [loss_tensor, y_tensor])
    return gradients[0] if len(gradients) == 1 else gradients


def reset_keras(delete_list = None, frac = 1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = frac)
    config = tf.ConfigProto(gpu_options=gpu_options)

    if delete_list is None:
        K.clear_session()
        s = tf.InteractiveSession(config = config)
        K.set_session(s)
    else:
        import gc
        K.clear_session()
        try:
            for d in delete_list:
                del d
        except:
            pass
        gc.collect()
        K.set_session(tf.Session(config = config))

def sample_input_for_loc_by_rd(
    indices_to_chgd,
    indices_to_unchgd,
    predictions = None, ys = None):
    """
    """
    num_chgd = len(indices_to_chgd)
    if num_chgd >= len(indices_to_unchgd): # no need to do any sampling
        return indices_to_chgd, indices_to_unchgd

    if predictions is None and ys is None:
        sampled_indices_to_unchgd = np.random.choice(indices_to_unchgd, num_chgd, replace = False)
        return indices_to_chgd, sampled_indices_to_unchgd
    else:
        _, sampled_indices_to_unchgd = sample_input_for_loc_sophis(
            indices_to_chgd,
            indices_to_unchgd,
            predictions, ys)

        return indices_to_chgd, sampled_indices_to_unchgd


def sample_input_for_loc_sophis(
    indices_to_chgd,
    indices_to_unchgd,
    predictions, ys):
    """
    prediction -> model ouput. Right before outputing as the final classification result
        from 0~len(indices_to_unchgd)-1, the results of unchagned
        from len(indices_to_unchgd)~end, the results of changed
    sample the indices to changed and unchanged behaviour later used for localisation
    """
    if len(ys.shape) > 1 and ys.shape[-1] > 1:
        pred_labels = np.argmax(predictions, axis = 1)
        y_labels = np.argmax(ys, axis = 1)
    else:
        pred_labels = np.round(predictions).flatten()
        y_labels = ys

    _indices = np.zeros(len(indices_to_unchgd) + len(indices_to_chgd))
    _indices[:len(indices_to_unchgd)] = indices_to_unchgd
    _indices[len(indices_to_unchgd):] = indices_to_chgd

    ###  checking  ###
    _indices_to_unchgd = np.where(pred_labels == y_labels)[0]; _indices_to_unchgd.sort()
    indices_to_unchgd = np.asarray(indices_to_unchgd); indices_to_unchgd.sort()
    _indices_to_chgd = np.where(pred_labels != y_labels)[0]; _indices_to_chgd.sort()
    indices_to_chgd = np.asarray(indices_to_chgd); indices_to_chgd.sort()

    assert all(indices_to_unchgd == _indices[_indices_to_unchgd])
    assert all(indices_to_chgd == np.sort(_indices[_indices_to_chgd]))
    ###  checking end  ###

    # here, only the labels of the ys (original labels) are considered
    uniq_labels = np.unique(y_labels[_indices_to_unchgd]); uniq_labels.sort()
    grouped_by_label = {uniq_label:[] for uniq_label in uniq_labels}
    for idx in _indices_to_unchgd:
        pred_label = pred_labels[idx]
        grouped_by_label[pred_label].append(idx)

    num_unchgd = len(indices_to_unchgd)
    num_chgd = len(indices_to_chgd)
    sampled_indices_to_unchgd = []
    num_total_sampled = 0
    for _,vs in grouped_by_label.items():
        num_sample = int(np.round(num_chgd * len(vs)/num_unchgd))
        if num_sample <= 0:
            num_sample = 1

        if num_sample > len(vs):
            num_sample = len(vs)

        sampled_indices_to_unchgd.extend(list(np.random.choice(vs, num_sample, replace = False)))
        num_total_sampled += num_sample

    #print ("Total number of sampled: {}".format(num_total_sampled))
    return indices_to_chgd, sampled_indices_to_unchgd

#@tf.function
def compute_FI_and_GL(
    X, y,
    indices_to_target,
    target_weights,
    is_multi_label = True,
    path_to_keras_model = None,
    federated = False,
    sample_weights = None,
    use_gradient_loss = True):
    """
    compute FL and GL for the given inputs
    """
    if len(indices_to_target) == 0:
        return {}

    ## Now, start localisation !!! ##
    from sklearn.preprocessing import Normalizer
    from collections.abc import Iterable
    norm_scaler = Normalizer(norm = "l1")
    total_cands = {}
    FIs = None; grad_scndcr = None

    # For FedRep: dictionaries for intermediate scores of FI
    avg_activations = {}
    avg_activations_den = {}
    output_grads = {}

    #t0 = time.time()
    ## slice inputs
    target_X = X[indices_to_target]
    target_y = y[indices_to_target]
    weights = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype = float).flatten()
        if len(weights) != len(target_X):
            weights = weights[indices_to_target]
        weights = _normalise_sample_weights(len(target_X), weights)

    # get loss func
    loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
    model = None
    for idx_to_tl, vs in target_weights.items():
        t1 = time.time()
        t_w, lname = vs
        #model = load_model(path_to_keras_model, compile = False)

        if "hydra" not in str(path_to_keras_model):
            model = load_model(path_to_keras_model, compile=False)
        else:
            model = load_model_from_h5(path_to_keras_model)
        if idx_to_tl == 0:
            # meaning the model doesn't specify the input layer explicitly
            prev_output = target_X
        else:
            prev_output = model.layers[idx_to_tl - 1].output
        layer_config = model.layers[idx_to_tl].get_config()

        if model_util.is_FC(lname):
            from_front = []
            if idx_to_tl == 0 or idx_to_tl - 1 == 0:
                prev_output = target_X
            else:
                t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
                prev_output = t_model.predict(target_X)
            if len(prev_output.shape) == 3:
                prev_output = prev_output.reshape(prev_output.shape[0], prev_output.shape[-1])

            for idx in tqdm(range(t_w.shape[-1])):
                assert int(prev_output.shape[-1]) == t_w.shape[0], "{} vs {}".format(
                    int(prev_output.shape[-1]), t_w.shape[0])

                output = np.multiply(prev_output, t_w[:,idx]) # -> shape = prev_output.shape
                output = np.abs(output)
                output = norm_scaler.fit_transform(output)
                #print("BEfore mean",output.shape)
                output = _weighted_mean(output, weights)
                #print(output.shape)
                from_front.append(output)

            from_front = np.asarray(from_front)


            from_front = from_front.T
            from_behind = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X,federated=federated, sample_weights = weights)

            #print ("shape", from_front.shape, from_behind.shape)
            #print(from_behind)
            FIs = from_front * from_behind
            print(f"FI shape {FIs.shape}")
            ############ FI end #########

            #mask = np.full(from_front.shape, 1)
            masked_behind = from_behind
            masked_front = from_front


            #assert np.isclose(masked_front*masked_behind,FIs).all(), "Masked metrics do not match non-masked when multiplied together"

            # Gradient
            grad_scndcr = None
            if use_gradient_loss:
                grad_scndcr = compute_gradient_to_loss(
                    path_to_keras_model, idx_to_tl, target_X, target_y, loss_func=loss_func,federated=federated)
            # G end
        elif model_util.is_C2D(lname):
            is_channel_first = layer_config['data_format'] == 'channels_first'
            if idx_to_tl == 0 or idx_to_tl - 1 == 0:
                prev_output_v = target_X
            else:
                t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
                prev_output_v = t_model.predict(target_X)
            tr_prev_output_v = np.moveaxis(prev_output_v, [1,2,3],[3,1,2]) if is_channel_first else prev_output_v
            #print(tr_prev_output_v.shape)

            kernel_shape = t_w.shape[:2]
            strides = layer_config['strides']
            padding_type =  layer_config['padding']
            if padding_type == 'valid':
                paddings = [0,0]
            else:
                if padding_type == 'same':
                    #P = ((S-1)*W-S+F)/2
                    true_ws_shape = [t_w.shape[0], t_w.shape[-1]] # Channel_in, Channel_out
                    paddings = [int(((strides[i]-1)*true_ws_shape[i]-strides[i]+kernel_shape[i])/2) for i in range(2)]
                elif not isinstance(padding_type, str) and isinstance(padding_type, Iterable): # explicit paddings given
                    paddings = list(padding_type)
                    if len(paddings) == 1:
                        paddings = [paddings[0], paddings[0]]
                else:
                    print ("padding type: {} not supported".format(padding_type))
                    paddings = [0,0]
                    assert False

                # add padding
                if is_channel_first:
                    paddings_per_axis = [[0,0], [0,0], [paddings[0], paddings[0]], [paddings[1], paddings[1]]]
                else:
                    paddings_per_axis = [[0,0], [paddings[0], paddings[0]], [paddings[1], paddings[1]], [0,0]]

                tr_prev_output_v = np.pad(tr_prev_output_v, paddings_per_axis,
                    mode = 'constant', constant_values = 0) # zero-padding

            if is_channel_first:
                num_kernels = int(prev_output.shape[1]) # Channel_in
            else: # channels_last
                assert layer_config['data_format'] == 'channels_last', layer_config['data_format']
                num_kernels = int(prev_output.shape[-1]) # Channel_in
            assert num_kernels == t_w.shape[2], "{} vs {}".format(num_kernels, t_w.shape[2])
            #print ("t_w***", t_w.shape)

            # H x W
            if is_channel_first:
                # the last two (front two are # of inputs and # of kernels (Channel_in))
                input_shape = [int(v) for v in prev_output.shape[2:]]
            else:
                input_shape = [int(v) for v in prev_output.shape[1:-1]]

            # (W1−F+2P)/S+1, W1 = input volumne , F = kernel, P = padding
            n_mv_0 = int((input_shape[0] - kernel_shape[0] + 2 * paddings[0])/strides[0] + 1) # H_out
            n_mv_1 = int((input_shape[1] - kernel_shape[1] + 2 * paddings[1])/strides[1] + 1) # W_out

            n_output_channel = t_w.shape[-1]  # Channel_out
            from_front = []
            # move axis for easier computation
            for idx_ol in tqdm(range(n_output_channel)): # t_w.shape[-1]

                for i in range(n_mv_0): # H

                    for j in range(n_mv_1): # W
                        curr_prev_output_slice = tr_prev_output_v[:,i*strides[0]:i*strides[0]+kernel_shape[0],:,:]
                        curr_prev_output_slice = curr_prev_output_slice[:,:,j*strides[1]:j*strides[1]+kernel_shape[1],:]
                        output = curr_prev_output_slice * t_w[:,:,:,idx_ol]

                        # Fixed normalization of weight influence!
                        if is_channel_first:
                            raise Exception("Not implemented yet!")
                        else:
                            # Since it is channel_last, last three shape indexes are (width,height,channel_number)
                            # Change it to (channel_number,width,height) for simpler computattion
                            out_copy = np.moveaxis(output,-1,-3)

                            # Sum across width and height to compute the normalization denominator
                            sum_output = np.sum(np.sum(out_copy,axis=-1),axis=-1)
                            out_shape = output.shape
                            sum_shape = sum_output.shape

                            # This array will contain the normalization denominator for each activation
                            extended_sum = np.empty(out_shape)

                            for idx in itertools.product(*[range(s) for s in sum_shape]):
                                for i in range(3):
                                    for j in range(3):
                                        index = idx[:1] + (i,j) + idx[1:]
                                        try:
                                            extended_sum[index] = sum_output[idx]
                                        except:
                                            print(index,idx,extended_sum.shape)

                            #extended_sum = np.moveaxis(extended_sum,-1,-3)
                            normalized = output/extended_sum
                            #normalized = np.moveaxis(normalized, -1, -3)

                            assert normalized.shape == out_shape, f"Normalized shape does not match"

                            output = normalized

                            # Verify that the normalization is correct
                            #normalized = np.moveaxis(normalized,-1,-3)
                            #summed = np.sum(np.sum(normalized,axis=-1),axis=-1)
                            #assert (np.isclose(summed,1,atol=0.05)).all(), f"Error in Conv2D normalization!{summed}"

                        #sum_output = np.sum(np.abs(output))


                        #avg_output = np.mean(output,axis=0)
                        #output = output#/sum_output
                        output = np.nan_to_num(output, posinf = 0.)
                        output = _weighted_mean(output, weights)
                        from_front.append(output)

            from_front = np.asarray(from_front)

            #from_front.shape: [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in]
            if is_channel_first:
                from_front = from_front.reshape(
                    (n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
            else: # channels_last
                from_front = from_front.reshape(
                    (n_mv_0,n_mv_1,n_output_channel,kernel_shape[0],kernel_shape[1],int(prev_output.shape[-1])))

            # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
            # 	or [F1,F2,Channel_in, n_mv_0, n_mv_1,Channel_out]
            from_front = np.moveaxis(from_front, [0,1,2], [3,4,5])

            # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)]
            from_behind = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X, by_batch = False,federated=federated,
                sample_weights = weights)

            #t1 = time.time()
            # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1] (channels_firs)
            # or [F1,F2,Channel_in,n_mv_0, n_mv_1,Channel_out] (channels_last)
            #print(from_front.shape,from_behind.shape)
            FIs = from_front * from_behind


            # Artifically isolate from_behind and from_front
            masked_behind = from_behind #mask * from_behind
            masked_front=from_front

            # If some result is nan, replace with 0
            if np.isnan(FIs).any():
                FIs[np.isnan(FIs)] = 0
                masked_front[np.isnan(masked_front)] = 0

            #assert np.isclose(masked_front * masked_behind,FIs,rtol=1.e-2).all(), "Masked metrics do not match non-masked when multiplied together"

            #t2 = time.time()
            #print ('Time for multiplying front and behind results: {}'.format(t2 - t1))
            #FIs = np.mean(np.mean(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
            if is_channel_first:
                FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
            else:

                if not federated:
                    np.save("origAct",masked_front)
                    np.save("origOut",masked_behind)
                FIs = np.sum(np.sum(FIs, axis=-2), axis=-2)


            #from_behind = masked_behind
            #t3 = time.time()
            #print ('Time for computing mean for FIs: {}'.format(t3 - t2))
            ## Gradient
            # will be [F1, F2, Channel_in, Channel_out]
            grad_scndcr = None
            if use_gradient_loss:
                grad_scndcr = compute_gradient_to_loss(
                    path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = False, loss_func=loss_func,federated=federated)
            if (isinstance(grad_scndcr, np.ndarray) and grad_scndcr.shape[0] == 0)\
                    or (isinstance(grad_scndcr, list) and len(grad_scndcr) == 0):
                print("grad_scndcr type: " + str(type(grad_scndcr)))
                print("grad_scndcr shape: " + str(grad_scndcr.shape))
        elif model_util.is_LSTM(lname): #
            from scipy.special import expit as sigmoid
            num_weights = 2
            assert len(t_w) == num_weights, t_w
            # t_w_kernel:
            # (input_feature_size, 4 * num_units). t_w_recurr_kernel: (num_units, 4 * num_units)
            t_w_kernel, t_w_recurr_kernel = t_w

            # get the previous output, which will be the input of the lstm
            if model_util.is_Input(type(model.layers[idx_to_tl - 1]).__name__):
                prev_output = target_X
            else:
                # shape = (batch_size, time_steps, input_feature_size)
                t_model = Model(inputs = model.input, outputs = model.layers[idx_to_tl - 1].output)
                prev_output = t_model.predict(target_X)

            assert len(prev_output.shape) == 3, prev_output.shape
            num_features = prev_output.shape[-1] # the dimension of features that will be processed by the model

            num_units = t_w_recurr_kernel.shape[0]
            assert t_w_kernel.shape[0] == num_features, "{} (kernel) vs {} (input)".format(t_w_kernel.shape[0], num_features)

            # hidden state and cell state sequences computation
            # generate a temporary model that only contains the target lstm layer
            # but with the modification to return sequences of hidden and cell states
            temp_lstm_layer_inst = lstm_layer.LSTM_Layer(model.layers[idx_to_tl])
            hstates_sequence, cell_states_sequence = temp_lstm_layer_inst.gen_lstm_layer_from_another(prev_output)
            init_hstates, init_cell_states = lstm_layer.LSTM_Layer.get_initial_state(model.layers[idx_to_tl])
            if init_hstates is None:
                init_hstates = np.zeros((len(target_X), num_units))
            if init_cell_states is None:
                # shape = (batch_size, num_units)
                init_cell_states = np.zeros((len(target_X), num_units))

            # shape = (batch_size, time_steps + 1, num_units)
            hstates_sequence = np.insert(hstates_sequence, 0, init_hstates, axis = 1)
             # shape = (batch_size, time_steps + 1, num_units)
            cell_states_sequence = np.insert(cell_states_sequence, 0, init_cell_states, axis = 1)
            bias = model.layers[idx_to_tl].get_weights()[-1] # shape = (4 * num_units,)
            indices_to_each_gates = np.array_split(np.arange(num_units * 4), 4)

            ## prepare all the intermediate outputs and the variables that will be used later
            idx_to_input_gate = 0
            idx_to_forget_gate = 1
            idx_to_cand_gate = 2
            idx_to_output_gate = 3

            # for kenerl, weight shape = (input_feature_size, num_units)
            # and for recurrent, (num_units, num_units), bias (num_units)
            # and the shape of all the intermedidate outpu is "(batch_size, time_step, num_units)"

            # input
            t_w_kernel_I = t_w_kernel[:, indices_to_each_gates[idx_to_input_gate]]
            t_w_recurr_kernel_I = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_input_gate]]
            bias_I = bias[indices_to_each_gates[idx_to_input_gate]]
            I = sigmoid(np.dot(prev_output, t_w_kernel_I) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_I) + bias_I)

            # forget
            t_w_kernel_F = t_w_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
            t_w_recurr_kernel_F = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_forget_gate]]
            bias_F = bias[indices_to_each_gates[idx_to_forget_gate]]
            F = sigmoid(np.dot(prev_output, t_w_kernel_F) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_F) + bias_F)

            # cand
            t_w_kernel_C = t_w_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
            t_w_recurr_kernel_C = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_cand_gate]]
            bias_C = bias[indices_to_each_gates[idx_to_cand_gate]]
            C = np.tanh(np.dot(prev_output, t_w_kernel_C) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_C) + bias_C)

            # output
            t_w_kernel_O = t_w_kernel[:, indices_to_each_gates[idx_to_output_gate]]
            t_w_recurr_kernel_O = t_w_recurr_kernel[:, indices_to_each_gates[idx_to_output_gate]]
            bias_O = bias[indices_to_each_gates[idx_to_output_gate]]
            # shape = (batch_size, time_steps, num_units)
            O = sigmoid(np.dot(prev_output, t_w_kernel_O) + np.dot(hstates_sequence[:,:-1,:], t_w_recurr_kernel_O) + bias_O)

            # set arguments to compute forward impact for the neural weights from these four gates
            t_w_kernels = {
                'input':t_w_kernel_I, 'forget':t_w_kernel_F,
                'cand':t_w_kernel_C, 'output':t_w_kernel_O}
            t_w_recurr_kernels = {
                'input':t_w_recurr_kernel_I, 'forget':t_w_recurr_kernel_F,
                'cand':t_w_recurr_kernel_C, 'output':t_w_recurr_kernel_O}

            consts = {}
            consts['input'] = get_constants('input', F, I, C, O, cell_states_sequence)
            consts['forget'] = get_constants('forget', F, I, C, O, cell_states_sequence)
            consts['cand'] = get_constants('cand', F, I, C, O, cell_states_sequence)
            consts['output'] = get_constants('output', F, I, C, O, cell_states_sequence)

            # from_front's shape = (num_units, (num_features + num_units) * 4)
            # gate_orders = ['input', 'forget', 'cand', 'output']
            from_front, gate_orders  = lstm_local_front_FI_for_target_all(
                prev_output, hstates_sequence[:,:-1,:], num_units,
                t_w_kernels, t_w_recurr_kernels, consts)

            from_front = from_front.T # ((num_features + num_units) * 4, num_units)
            N_k_rk_w = int(from_front.shape[0]/4)
            assert N_k_rk_w == num_features + num_units, "{} vs {}".format(N_k_rk_w, num_features + num_units)

            ## from behind
            from_behind = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X, by_batch = True, sample_weights = weights) # shape = (num_units,)

            #t1 = time.time()
            # shape = (N_k_rk_w, num_units)
            FIs_combined = from_front * from_behind
            #print ("Shape", from_behind.shape, FIs_combined.shape)
            #t2 = time.time()
            #print ('Time for multiplying front and behind results: {}'.format(t2 - t1))

            # reshaping
            FIs_kernel = np.zeros(t_w_kernel.shape) # t_w_kernel's shape (num_features, num_units * 4)
            FIs_recurr_kernel = np.zeros(t_w_recurr_kernel.shape) # t_w_recurr_kernel's shape (num_units, num_units * 4)
            # from (4 * N_k_rk_w, num_units) to 4 * (N_k_rk_w, num_units)
            for i, FI_p_gate in enumerate(np.array_split(FIs_combined, 4, axis = 0)):
                # FI_p_gate's shape = (N_k_rk_w, num_units)
                # 	-> will divided into (num_features, num_units) & (num_units, num_units)
                # local indices that will split FI_p_gate (shape = (N_k_rk_w, num_units))
                # since we append the weights in order of a kernel weight and a recurrent kernel weight
                indices_to_features = np.arange(num_features)
                indices_to_units = np.arange(num_units) + num_features
                #FIs_kernel[indices_to_features + (i * N_k_rk_w)]
                # = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
                #FIs_recurr_kernel[indices_to_units + (i * N_k_rk_w)]
                # = FI_p_gate[indices_to_units] # shape = (num_units, num_units)
                FIs_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_features] # shape = (num_features, num_units)
                FIs_recurr_kernel[:, i * num_units:(i+1) * num_units] = FI_p_gate[indices_to_units] # shape = (num_units, num_units)

            #t3 =time.time()
            FIs = [FIs_kernel, FIs_recurr_kernel] # [(num_features, num_units*4), (num_units, num_units*4)]
            #print ('Time for formatting: {}'.format(t3 - t2))

            ## Gradient
            grad_scndcr = None
            if use_gradient_loss:
                grad_scndcr = compute_gradient_to_loss(
                    path_to_keras_model, idx_to_tl, target_X, target_y, by_batch = True, loss_func = loss_func)

        else:
            print ("Currenlty not supported: {}. (shoulde be filtered before)".format(lname))
            import sys; sys.exit()

        #t2 = time.time()
        #print ("Time for computing cost for the {} layer: {}".format(idx_to_tl, t2 - t1))
        if not model_util.is_LSTM(target_weights[idx_to_tl][1]): # only one weight variable to process
            if isinstance(grad_scndcr, list):
                grad_scndcr = np.array(grad_scndcr)
                print("grad scndcr was a list, now has shape:")
                print(grad_scndcr.shape)
            if isinstance(FIs, list):
                FIs = np.array(FIs)
                print("FIs was a list, now has shape:")
                print(FIs.shape)

            if federated:
                if grad_scndcr is not None:
                    pairs = np.asarray([grad_scndcr.flatten()]).T
                else:
                    pairs = np.asarray([FIs.flatten()]).T
            else:
                if grad_scndcr is None:
                    pairs = np.asarray([FIs.flatten()]).T
                else:
                    pairs = np.asarray([grad_scndcr.flatten(), FIs.flatten()]).T
            total_cands[idx_to_tl] = {'shape':FIs.shape, 'costs':pairs}
        else: # currently, all of them go into here
            total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
            pairs = []
            for _FIs, _grad_scndcr in zip(FIs, grad_scndcr if grad_scndcr is not None else [None]*len(FIs)):

                if federated:
                    if _grad_scndcr is not None:
                        pairs = np.asarray(_grad_scndcr.flatten()).T
                    else:
                        pairs = np.asarray(_FIs.flatten()).T
                else:
                    if _grad_scndcr is None:
                        pairs = np.asarray([_FIs.flatten()]).T
                    else:
                        pairs = np.asarray([_grad_scndcr.flatten(), _FIs.flatten()]).T
                print(f"APPENDED SHAPE {_FIs.shape}")
                total_cands[idx_to_tl]['shape'].append(_FIs.shape)
                total_cands[idx_to_tl]['costs'].append(pairs)

        # Save average activation and output gradients
        avg_activations[idx_to_tl] = {'shape':masked_front.shape,'costs':masked_front}
        output_grads[idx_to_tl] = {'shape':masked_behind.shape,'costs':masked_behind}

    #t3 = time.time()
    #print ("Time for computing total costs: {}".format(t3 - t0))

    if not federated:
        return total_cands
    else:
        print(total_cands)
        return total_cands, avg_activations, output_grads


def compute_output_per_w(x, h, t_w_kernel, t_w_recurr_kernel, const, with_norm = False): 
    """
    A slice for a single neuron (unit or lstm cell)
    x = (batch_size, time_steps, num_features)
    h = (batch_size, time_steps, num_units)
    t_w_kernel = (num_features,)
    t_w_recurr_kernel = (num_units,)
    consts = (batch_size, time_steps) -> the value that is multiplied in the final state computation
    Return the product of the multiplication of weights and input for each unit (i.e., each LSTM cell)
    -> meaning the direct front impact computed per neural weights
    """
    from sklearn.preprocessing import Normalizer
    norm_scaler = Normalizer(norm = "l1")

    # here, "multiply" instead of dot, b/c we want to get the output of each neural weight (not the final one)
    out_kernel = x * t_w_kernel #np.multiply(x, t_w_kernel) # shape = (batch_size, time_steps, num_features)
    if h is not None: # None -> we will skip the weights for hiddens states
        out_recurr_kernel = h * t_w_recurr_kernel # shape = (batch_size, time_steps, num_units)
        out = np.append(out_kernel, out_recurr_kernel, axis = -1) # shape:(batch_size,time_steps,(num_features+num_units))
    else:
        out = out_kernel # shape = (batch_size, time_steps, num_features)

    # normalise
    out = np.abs(out)
    if with_norm:
        original_shape = out.shape
        out = norm_scaler.fit_transform(out.flatten().reshape(1,-1)).reshape(-1,)
        out = out.reshape(original_shape)

    # N = num_features or num_features + num_units
    out = np.einsum('ijk,ij->ijk', out, const) # shape = (batch_size, time_steps, N)
    return out


def get_constants(gate, F, I, C, O, cell_states):
    """
    """
    if gate == 'input':
        return np.multiply(O, np.divide(
            C, cell_states[:,1:,:],
            out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
    elif gate == 'forget':
        return np.multiply(O, np.divide(
            cell_states[:,:-1,:], cell_states[:,1:,:],
            out = np.zeros_like(cell_states[:,:-1,:]), where = cell_states[:,1:,:] != 0))
    elif gate == 'cand':
        return np.multiply(O, np.divide(
            I, cell_states[:,1:,:],
            out = np.zeros_like(C), where = cell_states[:,1:,:] != 0))
    else: # output
        return np.tanh(cell_states[:,1:,:])


def lstm_local_front_FI_for_target_all(
    x, h, num_units,
    t_w_kernels, t_w_recurr_kernels, consts,
    gate_orders = ['input', 'forget', 'cand', 'output']):
    """
    x = previous output
    h = hidden state (-> should be computed using the layer)
    t_w_kernels / t_w_recurr_kernels / consts:
        a group of neural weights that should be taken into account when measuring the impact.
        arg consts is the corresponding group of constants that will be multiplied
        to each nueral weight's output, respectively
    """
    from sklearn.preprocessing import Normalizer
    norm_scaler = Normalizer(norm = "l1")
    from_front = []
    for idx_to_unit in tqdm(range(num_units)):
        out_combined = None
        for gate in gate_orders:
            # out's shape, (batch_size, time_steps, (num_features + num_units))
            # since the weights of each gate are added with the weights of the other gates, normalise later
            out = compute_output_per_w(
                x, h,
                t_w_kernels[gate][:,idx_to_unit],
                t_w_recurr_kernels[gate][:,idx_to_unit],
                consts[gate][...,idx_to_unit],
                with_norm = False)

            if out_combined is None:
                out_combined = out
            else:
                out_combined = np.append(out_combined, out, axis = -1)

        # the shape of out_combined =>
        # 	(batch_size, time_steps, 4 * (num_features + num_units)) (since this is per unit)
        # here, keep in mind that we have to use a scaler on the current out_combined
        # (for instance, divide by the final output (the last hidden state won't work here anymore,
        # as the summation of the current value differs from the original due to
        # the absence of act and the scaling in the middle, etc.)
        original_shape = out_combined.shape
        # normalised
        scaled_out_combined = norm_scaler.fit_transform(np.abs(out_combined).flatten().reshape(1,-1))
        scaled_out_combined = scaled_out_combined.reshape(original_shape)
        # mean out_combined's shape: ((num_features + num_units) * 4,)
        # for each neural weight, the average over both time step and the batch
        avg_scaled_out_combined = np.mean(
            scaled_out_combined.reshape(-1, scaled_out_combined.shape[-1]), axis = 0)
        from_front.append(avg_scaled_out_combined)

    # from_front's shape = (num_units, (num_features + num_units) * 4)
    from_front = np.asarray(from_front)
    print ("For lstm's front part of FI: {}".format(from_front.shape))
    return from_front, gate_orders


def localise_by_chgd_unchgd(
    X, y,
    indices_to_chgd,
    indices_to_unchgd,
    target_weights,
    path_to_keras_model = None,
    is_multi_label = True):
    """
    Find those likely to be highly influential to the changed behaviour
    while less influential to the unchanged behaviour
    """
    from collections.abc import Iterable
    #loc_start_time = time.time()
    #print ("Layers to inspect", list(target_weights.keys()))
    # compute FI and GL with changed inputs
    #target_weights = {k:target_weights[k] for k in [2]}
    total_cands_chgd = compute_FI_and_GL(
        X, y,
        indices_to_chgd,
        target_weights,
        is_multi_label = is_multi_label,
        path_to_keras_model = path_to_keras_model)

    # compute FI and GL with unchanged inputs
    total_cands_unchgd = compute_FI_and_GL(
        X, y,
        indices_to_unchgd,
        target_weights,
        is_multi_label = is_multi_label,
        path_to_keras_model = path_to_keras_model)

    indices_to_tl = list(total_cands_chgd.keys())
    costs_and_keys = []; indices_to_nodes = []
    shapes = {}
    for idx_to_tl in tqdm(indices_to_tl):
        if not model_util.is_LSTM(target_weights[idx_to_tl][1]): # we have only one weight to process
            #assert not isinstance(
            #	total_cands_unchgd[idx_to_tl]['shape'], Iterable),
            # 	type(total_cands_unchgd[idx_to_tl]['shape'])
            cost_from_chgd = total_cands_chgd[idx_to_tl]['costs']
            cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs']
            ## key: more influential to changed behaviour and less influential to unchanged behaviour
            costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
            #costs_combined = cost_from_chgd
            shapes[idx_to_tl] = total_cands_chgd[idx_to_tl]['shape']

            for i,c in enumerate(costs_combined):
                costs_and_keys.append(([idx_to_tl, i], c))
                indices_to_nodes.append([idx_to_tl, np.unravel_index(i, shapes[idx_to_tl])])
        else: #
            #assert isinstance(
            #	total_cands_unchgd[idx_to_tl]['shape'], Iterable),
            #	type(total_cands_unchgd[idx_to_tl]['shape'])
            num = len(total_cands_unchgd[idx_to_tl]['shape'])
            shapes[idx_to_tl] = []
            for idx_to_pair in range(num):
                cost_from_chgd = total_cands_chgd[idx_to_tl]['costs'][idx_to_pair]
                cost_from_unchgd = total_cands_unchgd[idx_to_tl]['costs'][idx_to_pair]
                costs_combined = cost_from_chgd/(1. + cost_from_unchgd) # shape = (N,2)
                shapes[idx_to_tl].append(total_cands_chgd[idx_to_tl]['shape'][idx_to_pair])

                for i,c in enumerate(costs_combined):
                    costs_and_keys.append(([(idx_to_tl, idx_to_pair), i], c))
                    indices_to_nodes.append(
                        [(idx_to_tl, idx_to_pair), np.unravel_index(i, shapes[idx_to_tl][idx_to_pair])])

    costs = np.asarray([vs[1] for vs in costs_and_keys])
    #t4 = time.time()
    _costs = costs.copy()
    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0 # Next index in the is_efficient array to search for
    while next_point_index < len(_costs):
        nondominated_point_mask = np.any(_costs > _costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        _costs = _costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    pareto_front = [tuple(v) for v in np.asarray(indices_to_nodes, dtype = object)[is_efficient]]
    #t5 = time.time()
    #print ("Time for computing the pareto front: {}".format(t5 - t4))
    #loc_end_time = time.time()
    #print ("Time for total localisation: {}".format(loc_end_time - loc_start_time))
    return pareto_front, costs_and_keys


def localise_by_qexec_qres_sbfl(
    X, y,
    predictions,
    target_weights,
    path_to_keras_model = None,
    is_multi_label = True,
    eps = 1e-12):
    """
    Localise using forward feedback (FI) as participation and SBFL-style success/failure scores.

    Success score: predicted probability of the (correct) class when prediction is correct.
    Failure score: 1 - probability of the wrongly predicted class when prediction is incorrect.
    Suspiciousness per weight: fail_participation / (fail_participation + pass_participation).
    """
    if predictions.ndim == 1:
        pred_labels = np.round(predictions).astype(int)
        pred_confidence = predictions.flatten()
    else:
        pred_labels = np.argmax(predictions, axis = 1)
        pred_confidence = predictions[np.arange(len(pred_labels)), pred_labels]

    if y.ndim > 1:
        true_labels = np.argmax(y, axis = 1)
    else:
        true_labels = y

    correct_mask = pred_labels == true_labels
    success_scores = np.where(correct_mask, pred_confidence, 0.0)
    failure_scores = np.where(~correct_mask, 1.0 - pred_confidence, 0.0)

    indices_to_pass = np.where(success_scores > 0)[0]
    indices_to_fail = np.where(failure_scores > 0)[0]

    fail_cands = compute_FI_and_GL(
        X, y, indices_to_fail, target_weights,
        is_multi_label = is_multi_label,
        path_to_keras_model = path_to_keras_model,
        federated = False,
        sample_weights = failure_scores,
        use_gradient_loss = False)
    pass_cands = compute_FI_and_GL(
        X, y, indices_to_pass, target_weights,
        is_multi_label = is_multi_label,
        path_to_keras_model = path_to_keras_model,
        federated = False,
        sample_weights = success_scores,
        use_gradient_loss = False)

    indices_to_nodes = []
    costs_and_keys = []

    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {'costs':[], 'shape':[]}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {'costs':[], 'shape':[]}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get('shape') or pass_entry.get('shape') or t_w.shape
            fail_costs = fail_entry.get('costs', None)
            pass_costs = pass_entry.get('costs', None)

            if fail_costs is None or len(np.asarray(fail_costs)) == 0:
                fail_vals = np.zeros(np.prod(target_shape))
            else:
                fail_vals = np.asarray(fail_costs).reshape(-1)
            if pass_costs is None or len(np.asarray(pass_costs)) == 0:
                pass_vals = np.zeros(np.prod(target_shape))
            else:
                pass_vals = np.asarray(pass_costs).reshape(-1)

            if fail_vals.shape != pass_vals.shape:
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros_like(pass_vals)
                elif pass_vals.shape[0] == 0:
                    pass_vals = np.zeros_like(fail_vals)

            suspiciousness = fail_vals/(fail_vals + pass_vals + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
                indices_to_nodes.append([idx_to_tl, np.unravel_index(i, target_shape)])
        else:
            target_shapes = fail_entry.get('shape') or pass_entry.get('shape') or [w.shape for w in t_w]
            fail_costs_list = fail_entry.get('costs', [])
            pass_costs_list = pass_entry.get('costs', [])

            for idx_to_w, shape in enumerate(target_shapes):
                _fail_costs = fail_costs_list[idx_to_w] if idx_to_w < len(fail_costs_list) else []
                _pass_costs = pass_costs_list[idx_to_w] if idx_to_w < len(pass_costs_list) else []
                fail_vals = np.asarray(_fail_costs).reshape(-1)
                pass_vals = np.asarray(_pass_costs).reshape(-1)
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros(np.prod(shape))
                if pass_vals.shape[0] == 0:
                    pass_vals = np.zeros(np.prod(shape))

                suspiciousness = fail_vals/(fail_vals + pass_vals + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))
                    indices_to_nodes.append(
                        [(idx_to_tl, idx_to_w), np.unravel_index(local_i, shape)])

    sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
    return sorted_costs_and_keys


def localise_by_gradient(
    X, y,
    indices_to_chgd,
    indices_to_unchgd,
    target_weights,
    path_to_keras_model = None,
    is_multi_label = True):
    """
    localise using chgd & unchgd
    """
    from collections.abc import Iterable

    total_cands = {}
    # set loss func
    loss_func = model_util.get_loss_func(is_multi_label = is_multi_label)
    ## slice inputs
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        #print ("targeting layer {} ({})".format(idx_to_tl, lname))
        if model_util.is_C2D(lname) or model_util.is_FC(lname): # either FC or C2D
            # for changed inputs
            grad_scndcr_for_chgd = compute_gradient_to_loss(
                path_to_keras_model, idx_to_tl, X[indices_to_chgd], y[indices_to_chgd],
                loss_func = loss_func, by_batch = True)
            # for unchanged inputs
            grad_scndcr_for_unchgd = compute_gradient_to_loss(
                path_to_keras_model, idx_to_tl, X[indices_to_unchgd], y[indices_to_unchgd],
                loss_func = loss_func, by_batch = True)

            assert t_w.shape == grad_scndcr_for_chgd.shape, "{} vs {}".format(t_w.shape, grad_scndcr_for_chgd.shape)
            total_cands[idx_to_tl] = {
                'shape':grad_scndcr_for_chgd.shape,
                'costs':grad_scndcr_for_chgd.flatten()/(1.+grad_scndcr_for_unchgd.flatten())}
        elif model_util.is_LSTM(lname):
            # for changed inputs
            grad_scndcr_for_chgd = compute_gradient_to_loss(
                path_to_keras_model, idx_to_tl,
                X[indices_to_chgd], y[indices_to_chgd],
                loss_func = loss_func, by_batch = True)
            # for unchanged inptus
            grad_scndcr_for_unchgd = compute_gradient_to_loss(
                path_to_keras_model, idx_to_tl,
                X[indices_to_unchgd], y[indices_to_unchgd],
                loss_func = loss_func, by_batch = True)

            # check the shape of kernel (index = 0) and recurrent kernel (index =1) weights
            assert t_w[0].shape == grad_scndcr_for_chgd[0].shape, "{} vs {}".format(t_w[0].shape, grad_scndcr_for_chgd[0].shape)
            assert t_w[1].shape == grad_scndcr_for_chgd[1].shape, "{} vs {}".format(t_w[1].shape, grad_scndcr_for_chgd[1].shape)

            # generate total candidates
            total_cands[idx_to_tl] = {'shape':[], 'costs':[]}
            for _grad_scndr_chgd, _grad_scndr_unchgd in zip(grad_scndcr_for_chgd, grad_scndcr_for_unchgd):
                #_grad_scndr_chgd & _grad_scndr_unchgd -> can be for either kernel or recurrent kernel
                _costs = _grad_scndr_chgd.flatten()/(1. + _grad_scndr_unchgd.flatten())
                total_cands[idx_to_tl]['shape'].append(_grad_scndr_chgd.shape)
                total_cands[idx_to_tl]['costs'].append(_costs)
        else:
            print ("{} not supported yet".format(lname))
            assert False

    indices_to_tl = list(total_cands.keys())
    costs_and_keys = []
    for idx_to_tl in indices_to_tl:
        if not model_util.is_LSTM(target_weights[idx_to_tl][1]):
            for local_i,c in enumerate(total_cands[idx_to_tl]['costs']):
                cost_and_key = ([idx_to_tl, np.unravel_index(local_i, total_cands[idx_to_tl]['shape'])], c)
                costs_and_keys.append(cost_and_key)
        else:
            num = len(total_cands[idx_to_tl]['shape'])
            for idx_to_w in range(num):
                for local_i, c in enumerate(total_cands[idx_to_tl]['costs'][idx_to_w]):
                    cost_and_key = (
                        [(idx_to_tl, idx_to_w),
                        np.unravel_index(local_i, total_cands[idx_to_tl]['shape'][idx_to_w])], c)
                    costs_and_keys.append(cost_and_key)

    sorted_costs_and_keys = sorted(costs_and_keys, key = lambda vs:vs[1], reverse = True)
    return sorted_costs_and_keys


def localise_by_random_selection(number_of_place_to_fix, target_weights):
    """
    randomly select places to fix
    """
    from collections.abc import Iterable

    total_indices = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        if not model_util.is_LSTM(lname):
            l_indices = list(np.ndindex(t_w.shape))
            total_indices.extend(list(zip([idx_to_tl] * len(l_indices), l_indices)))
        else: # to handle the layers with more than one weights (e.g., LSTM)
            for idx_to_w, a_t_w in enumerate(t_w):
                l_indices = list(np.ndindex(a_t_w.shape))
                total_indices.extend(list(zip([(idx_to_tl, idx_to_w)] * len(l_indices), l_indices)))

    np.random.shuffle(total_indices)
    if number_of_place_to_fix > 0 and number_of_place_to_fix < len(total_indices):
        selected_indices = np.random.choice(
            np.arange(len(total_indices)), number_of_place_to_fix, replace = False)
        indices_to_places_to_fix = [total_indices[idx] for idx in selected_indices]
    else:
        indices_to_places_to_fix = total_indices

    return indices_to_places_to_fix


# --- SBFL-style localisers (added) ---

def _get_layer_output(model, layer_idx, X):
    from tensorflow.keras.models import Model
    if not hasattr(_get_layer_output, "_cache"):
        _get_layer_output._cache = {}
    cache = _get_layer_output._cache
    if layer_idx not in cache:
        cache[layer_idx] = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    return cache[layer_idx].predict(X)


def _coverage_per_unit(model, layer_idx, X, sample_weights, activation_threshold=0.1):
    """
    Return coverage per output unit/channel for a given layer, weighted by sample_weights.
    For Dense: shape (out_units,)
    For Conv2D: shape (out_channels,)
    """
    out = _get_layer_output(model, layer_idx, X)
    act = np.abs(out)
    # reduce spatial dims for conv
    lname = type(model.layers[layer_idx]).__name__
    if model_util.is_FC(lname):
        act_unit = np.max(act, axis=0)  # (out_units,)
    elif model_util.is_C2D(lname):
        # assume channels_last
        act_unit = np.max(act, axis=(0, 1, 2))  # (out_channels,)
    else:
        return None
    max_abs = np.max(act_unit) if act_unit.size else 0.0
    if max_abs > 0:
        act_unit = act_unit / max_abs
    cov_unit = (act_unit >= activation_threshold).astype(float)
    # weight by sample_weights (scalar per sample); here act_unit already aggregated over samples,
    # so just scale coverage by mean weight
    weight_scalar = float(np.mean(sample_weights)) if sample_weights is not None and len(sample_weights) else 1.0
    return cov_unit * weight_scalar


def _suspicious_from_cov(fail_cov, pass_cov, target_shape, eps=1e-12):
    fail_vals = np.repeat(fail_cov, int(np.prod(target_shape[:-1]))) if len(target_shape) > 2 else fail_cov
    pass_vals = np.repeat(pass_cov, int(np.prod(target_shape[:-1]))) if len(target_shape) > 2 else pass_cov
    total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
    sus = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
    return sus


def _build_results_from_cov(layer_idx, lname, t_w, fail_cov, pass_cov, eps=1e-12):
    costs_and_keys = []
    indices_to_nodes = []
    if model_util.is_FC(lname):
        out_dim = t_w.shape[-1]
        in_dim = t_w.shape[0]
        fail_vals = np.repeat(fail_cov, in_dim).reshape(out_dim, in_dim).T.flatten()
        pass_vals = np.repeat(pass_cov, in_dim).reshape(out_dim, in_dim).T.flatten()
        target_shape = t_w.shape
    elif model_util.is_C2D(lname):
        out_dim = t_w.shape[-1]
        repeat_size = int(np.prod(t_w.shape[:-1]))
        fail_vals = np.repeat(fail_cov, repeat_size)
        pass_vals = np.repeat(pass_cov, repeat_size)
        target_shape = t_w.shape
    else:
        return costs_and_keys
    total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
    suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
    for i, sus in enumerate(suspiciousness):
        costs_and_keys.append(([layer_idx, i], sus))
        indices_to_nodes.append([layer_idx, np.unravel_index(i, target_shape)])
    return costs_and_keys


def _extract_costs(entry):
    costs = entry.get("costs", None)
    if costs is None:
        return None
    arr = np.asarray(costs)
    if arr.ndim > 1 and arr.shape[1] >= 2:
        # assume last column corresponds to FI
        return arr[:, -1]
    return arr.reshape(-1)


def _extract_costs_list(entry_list):
    res = []
    for c in entry_list:
        arr = np.asarray(c)
        if arr.ndim > 1 and arr.shape[1] >= 2:
            res.append(arr[:, -1])
        else:
            res.append(arr.reshape(-1))
    return res


def localise_by_qexec_qres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1):
    """
    Quantised execution (FI-based) + quantised result (probability) SBFL (Ochiai).
    """
    if predictions.ndim == 1:
        pred_labels = np.round(predictions).astype(int)
        pred_confidence = predictions.flatten()
    else:
        pred_labels = np.argmax(predictions, axis=1)
        pred_confidence = predictions[np.arange(len(pred_labels)), pred_labels]
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y

    correct_mask = pred_labels == true_labels
    success_scores = np.where(correct_mask, pred_confidence, 0.0)
    failure_scores = np.where(~correct_mask, pred_confidence, 0.0)

    idx_pass = np.where(success_scores > 0)[0]
    idx_fail = np.where(failure_scores > 0)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        rng = np.random.default_rng()
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)

    fail_cands = compute_FI_and_GL(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=failure_scores[idx_fail] if len(idx_fail) else None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=success_scores[idx_pass] if len(idx_pass) else None,
        use_gradient_loss=False)

    costs_and_keys = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape") or t_w.shape
            fail_vals = _extract_costs(fail_entry)
            pass_vals = _extract_costs(pass_entry)
            if fail_vals is None:
                fail_vals = np.zeros(np.prod(target_shape))
            if pass_vals is None:
                pass_vals = np.zeros(np.prod(target_shape))
            if fail_vals.shape != pass_vals.shape:
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros_like(pass_vals)
                elif pass_vals.shape[0] == 0:
                    pass_vals = np.zeros_like(fail_vals)
            total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
            suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
        else:
            shapes = fail_entry.get("shape") or pass_entry.get("shape") or [w.shape for w in t_w]
            fail_list = fail_entry.get("costs", [])
            pass_list = pass_entry.get("costs", [])
            fail_vals_list = _extract_costs_list(fail_list) if fail_list else []
            pass_vals_list = _extract_costs_list(pass_list) if pass_list else []
            for idx_to_w, shape in enumerate(shapes):
                fail_vals = fail_vals_list[idx_to_w] if idx_to_w < len(fail_vals_list) else np.zeros(np.prod(shape))
                pass_vals = pass_vals_list[idx_to_w] if idx_to_w < len(pass_vals_list) else np.zeros(np.prod(shape))
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros(np.prod(shape))
                if pass_vals.shape[0] == 0:
                    pass_vals = np.zeros(np.prod(shape))
                total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
                suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)


def localise_by_qexec_bres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1):
    """
    Quantised execution (FI-based) + binary result (pass/fail=1).
    """
    if predictions.ndim == 1:
        pred_labels = np.round(predictions).astype(int)
    else:
        pred_labels = np.argmax(predictions, axis=1)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y
    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        rng = np.random.default_rng()
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)

    fail_cands = compute_FI_and_GL(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)

    costs_and_keys = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape") or t_w.shape
            fail_vals = _extract_costs(fail_entry)
            pass_vals = _extract_costs(pass_entry)
            if fail_vals is None:
                fail_vals = np.zeros(np.prod(target_shape))
            if pass_vals is None:
                pass_vals = np.zeros(np.prod(target_shape))
            if fail_vals.shape != pass_vals.shape:
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros_like(pass_vals)
                elif pass_vals.shape[0] == 0:
                    pass_vals = np.zeros_like(fail_vals)
            total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
            suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
        else:
            shapes = fail_entry.get("shape") or pass_entry.get("shape") or [w.shape for w in t_w]
            fail_list = fail_entry.get("costs", [])
            pass_list = pass_entry.get("costs", [])
            fail_vals_list = _extract_costs_list(fail_list) if fail_list else []
            pass_vals_list = _extract_costs_list(pass_list) if pass_list else []
            for idx_to_w, shape in enumerate(shapes):
                fail_vals = fail_vals_list[idx_to_w] if idx_to_w < len(fail_vals_list) else np.zeros(np.prod(shape))
                pass_vals = pass_vals_list[idx_to_w] if idx_to_w < len(pass_vals_list) else np.zeros(np.prod(shape))
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros(np.prod(shape))
                if pass_vals.shape[0] == 0:
                    pass_vals = np.zeros(np.prod(shape))
                total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
                suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)


def localise_by_bexec_qres_guider(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1):
    """
    Binary execution (activation coverage) + quantised result (probability).
    """
    if predictions.ndim == 1:
        pred_labels = np.round(predictions).astype(int)
        pred_confidence = predictions.flatten()
    else:
        pred_labels = np.argmax(predictions, axis=1)
        pred_confidence = predictions[np.arange(len(pred_labels)), pred_labels]
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y
    correct_mask = pred_labels == true_labels
    success_scores = np.where(correct_mask, pred_confidence, 0.0)
    failure_scores = np.where(~correct_mask, pred_confidence, 0.0)

    idx_pass = np.where(success_scores > 0)[0]
    idx_fail = np.where(failure_scores > 0)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        rng = np.random.default_rng()
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)

    if "hydra" not in str(path_to_keras_model):
        model = load_model(path_to_keras_model, compile=False)
    else:
        model = load_model_from_h5(path_to_keras_model)

    costs_and_keys = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_cov = _coverage_per_unit(model, idx_to_tl, X[idx_fail], failure_scores[idx_fail] if len(idx_fail) else None, activation_threshold)
        pass_cov = _coverage_per_unit(model, idx_to_tl, X[idx_pass], success_scores[idx_pass] if len(idx_pass) else None, activation_threshold)
        if fail_cov is None and pass_cov is None:
            continue
        if fail_cov is None:
            fail_cov = np.zeros_like(pass_cov)
        if pass_cov is None:
            pass_cov = np.zeros_like(fail_cov)
        costs_and_keys.extend(_build_results_from_cov(idx_to_tl, lname, t_w, fail_cov, pass_cov, eps))

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)
