"""
Localise faults in offline for any faults
"""
# 本文件核心：实现多种定位算法（FI/GL、SBFL、覆盖等）。
# 主要流程：选择目标层 -> 计算参与度/梯度 -> 生成可疑权重排序。

import itertools

import numpy as np
import time
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras.models import load_model, Model
import tensorflow.compat.v1.keras.backend as K
tf.disable_eager_execution()
from tqdm import tqdm
try:
    from . import lstm_layer
    from .utils import model_util
except ImportError:
    import lstm_layer
    from utils import model_util

from tensorflow.keras import layers
try:
    from tensorflow.keras.layers import CategoryEncoding
except ImportError:
    # TF1.x fallback for CategoryEncoding
    class CategoryEncoding(tf.keras.layers.Layer):
        def __init__(self, num_tokens, output_mode="one_hot", **kwargs):
            super().__init__(**kwargs)
            self.num_tokens = num_tokens
            self.output_mode = output_mode

        def call(self, inputs):
            return tf.one_hot(inputs, depth=self.num_tokens)

from pathlib import Path

# "divided by zeros" is handleded afterward
np.seterr(divide='ignore', invalid='ignore')

try:
    _tf_function = tf.function
except AttributeError:
    def _tf_function(fn):
        return fn




# 自定义层：根据门控输出对分支结果加权融合
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

    @_tf_function
    def inner_predict(self, data, num):
        """Gating branch based on gating layer."""
        tf_list = tf.math.argmax(data, 1)
        gate_out = CategoryEncoding(num_tokens=num, output_mode="one_hot")(tf_list)
        return gate_out

# 加载 hydra 模型（json 结构 + h5 权重）
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

# 选择需要定位的层及其权重（支持 Dense/Conv/LSTM）
def get_target_weights(model, path_to_keras_model, indices_to_target = None):
    """
    return indices to weight layers denoted by indices_to_target, or return all trainable layers

    :param indices_to_target: number of layers to grab as targets. Grab the last X ones that have weights
    """
    import re
    targeting_clname_pattns = ['Dense*', 'Conv*', '.*LSTM*', 'BatchNormalization*'] #if not target_all else None
    is_target = lambda clname,targets: (targets is None) or any([bool(re.match(t,clname)) for t in targets])

    if model is None:
        assert path_to_keras_model is not None

        if "hydra" not in str(path_to_keras_model):
            model = load_model(path_to_keras_model, compile=False)
        else:
            model = load_model_from_h5(path_to_keras_model)

    target_weights = {} # key = layer index, value: [weight value, layer name]
    if indices_to_target is not None:
        # indices_to_target 可以是 int（取最近的若干层）或 list/array（指定层索引）
        if isinstance(indices_to_target, (list, tuple, np.ndarray)):
            layer_indices = sorted(set(int(x) for x in np.asarray(indices_to_target).flatten()))
        else:
            num_layers = len(model.layers)
            layer_indices = list(range(num_layers-1, -1, -1))
        for i in layer_indices:
            layer = model.layers[i]
            ws = layer.get_weights()
            if len(ws) == 0:
                print("the target layer doesn't have weight")
                continue
            elif model_util.is_BatchNorm(type(layer).__name__):
                # Keep BN trainable params (gamma/beta); skip non-trainable running stats
                target_weights[i] = [ws[:2], type(layer).__name__]
            else:
                target_weights[i] = [ws[0], type(layer).__name__]
            if not isinstance(indices_to_target, (list, tuple, np.ndarray)) and len(target_weights) >= indices_to_target:
                print(f"out of {len(model.layers)} layers, we will localize weights in layers: {str(target_weights.keys())}")
                break
    else:
        for i, layer in enumerate(model.layers):
            class_name = type(layer).__name__
            if is_target(class_name, targeting_clname_pattns):
                ws = layer.get_weights()
                if len(ws): # has weight
                    if model_util.is_FC(class_name) or model_util.is_C2D(class_name):
                        target_weights[i] = [ws[0], type(layer).__name__]
                    elif model_util.is_BatchNorm(class_name):
                        target_weights[i] = [ws[:2], type(layer).__name__]
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


# 样本权重清洗（裁剪负数，不做归一化）
def _clean_sample_weights(num, sample_weights):
    """
    Validate/clean sample weights; returns None when weights are absent or all-zero.
    """
    if sample_weights is None:
        return None
    weights = np.asarray(sample_weights, dtype=float).flatten()
    assert len(weights) == num, f"Weight length {len(weights)} does not match number of samples {num}"
    weights = np.clip(weights, a_min=0.0, a_max=None)
    if np.sum(weights) == 0:
        return None
    return weights


# 样本权重归一化（和为 1；全零返回 None）
def _normalise_sample_weights(num, sample_weights):
    """
    Normalise sample weights to sum to 1; returns None when weights are absent or all-zero.
    """
    weights = _clean_sample_weights(num, sample_weights)
    if weights is None:
        return None
    return weights / np.sum(weights)


# 按样本权重做加权平均
def _weighted_mean(arr, weights):
    """
    Weighted mean over axis 0; defaults to arithmetic mean when weights are None.
    """
    if weights is None:
        return np.mean(arr, axis=0)
    return np.tensordot(weights, arr, axes=([0], [0]))


# L1-normalise FI values (vector or per-sample matrix).
def _normalise_fi_values(values, eps=1e-12):
    arr = np.asarray(values)
    if arr.size == 0:
        return arr
    if arr.ndim == 1:
        denom = np.sum(np.abs(arr))
        if denom <= eps:
            return arr
        return arr / denom
    if arr.ndim == 2:
        denom = np.sum(np.abs(arr), axis=1, keepdims=True)
        denom = np.where(denom > eps, denom, 1.0)
        return arr / denom
    denom = np.sum(np.abs(arr), axis=-1, keepdims=True)
    denom = np.where(denom > eps, denom, 1.0)
    return arr / denom


# 计算输出对目标层输出/权重的梯度（可按样本加权）
def compute_gradient_to_output(path_to_keras_model, 
    idx_to_target_layer, X,
    by_batch = False, on_weight = False, wo_reset = False,federated=False,
    sample_weights = None,
    normalise_sample_weights = True):
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

    if normalise_sample_weights:
        weights = _normalise_sample_weights(num, sample_weights)
    else:
        weights = _clean_sample_weights(num, sample_weights)

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
        # NOTE: normalization disabled for inspection
        # norm_gradient = norm_scaler.fit_transform(reshaped_gradient) # normalised
        norm_gradient = reshaped_gradient

        # If no sample weights provided, return per-sample gradients
        if weights is None:
            # Return [n_samples, ...] - per-sample normalized gradients
            ret_gradient = norm_gradient.reshape(gradient.shape)
        else:
            # Aggregate with weights and return single gradient vector
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


# 计算损失对目标层权重的梯度（GL）
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


# 重置 Keras/TensorFlow 会话，清理计算图与缓存
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

# 随机采样未变化样本，使 changed/unchanged 数量接近
def sample_input_for_loc_by_rd(
    indices_to_chgd,
    indices_to_unchgd,
    predictions = None, ys = None):
    """
    Down-sample changed/unchanged to the same size to avoid extreme imbalance.
    If predictions/ys are given, unchanged sampling keeps label distribution.
    """
    num_chgd = len(indices_to_chgd)
    num_unchgd = len(indices_to_unchgd)
    if num_chgd == 0 or num_unchgd == 0:
        return indices_to_chgd, indices_to_unchgd

    num_sample = min(num_chgd, num_unchgd)

    # sample changed
    if num_sample < num_chgd:
        sampled_indices_to_chgd = np.random.choice(indices_to_chgd, num_sample, replace=False)
    else:
        sampled_indices_to_chgd = indices_to_chgd

    # sample unchanged (optionally stratified by label)
    if predictions is None and ys is None:
        sampled_indices_to_unchgd = np.random.choice(indices_to_unchgd, num_sample, replace=False)
    else:
        # reuse sophis sampler to keep label distribution, then cap to num_sample
        _, cand_unchgd = sample_input_for_loc_sophis(
            indices_to_chgd,
            indices_to_unchgd,
            predictions, ys)
        if len(cand_unchgd) > num_sample:
            sampled_indices_to_unchgd = np.random.choice(cand_unchgd, num_sample, replace=False)
        else:
            sampled_indices_to_unchgd = cand_unchgd

    return sampled_indices_to_chgd, sampled_indices_to_unchgd


# 外层 min-min 采样：changed/unchanged 都下采样到最小值
def sample_input_for_loc_by_minmin(
    indices_to_chgd,
    indices_to_unchgd):
    """
    """
    num_chgd = len(indices_to_chgd)
    num_unchgd = len(indices_to_unchgd)
    if num_chgd == 0 or num_unchgd == 0:
        return indices_to_chgd, indices_to_unchgd

    num_sample = min(num_chgd, num_unchgd)
    if num_sample == num_chgd and num_sample == num_unchgd:
        return indices_to_chgd, indices_to_unchgd

    sampled_indices_to_chgd = np.random.choice(indices_to_chgd, num_sample, replace = False)
    sampled_indices_to_unchgd = np.random.choice(indices_to_unchgd, num_sample, replace = False)
    return sampled_indices_to_chgd, sampled_indices_to_unchgd


# 按类别比例采样未变化样本，保持分布一致性
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
# 核心：计算 FI（前向影响×后向梯度）与可选 GL
def compute_FI_and_GL(
    X, y,
    indices_to_target,
    target_weights,
    is_multi_label = True,
    path_to_keras_model = None,
    federated = False,
    sample_weights = None,
    normalise_sample_weights = True,
    use_gradient_loss = True,
    aggregate = True):
    """
    compute FL and GL for the given inputs

    Args:
        aggregate: If True (default), aggregate across samples. If False, preserve sample dimension.
    """
    if len(indices_to_target) == 0:
        return {}

    ## Now, start localisation !!! ##
    from sklearn.preprocessing import Normalizer
    from collections.abc import Iterable
    norm_scaler = Normalizer(norm = "l1")  #归一化处理
    total_cands = {}
    FIs = None; grad_scndcr = None  #最终输出可疑权重字典 

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
        if normalise_sample_weights:
            weights = _normalise_sample_weights(len(target_X), weights)
        else:
            weights = _clean_sample_weights(len(target_X), weights)

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

                output = np.multiply(prev_output, t_w[:,idx]) # prev_output: 上一层的激活值 (Activation)# t_w[:,idx]: 当前神经元的权重 (Weight)  # -> shape = prev_output.shape
                output = np.abs(output)
                # NOTE: normalization disabled for inspection
                # output = norm_scaler.fit_transform(output)
                if aggregate:
                    #print("BEfore mean",output.shape)
                    output = _weighted_mean(output, weights)
                    #print(output.shape)

                from_front.append(output)

            from_front = np.asarray(from_front)

            if aggregate:
                from_front = from_front.T  # [n_features, n_neurons]
            else:
                # from_front 当前是 [n_neurons, n_samples, n_features]
                from_front = np.transpose(from_front, (1, 2, 0))  # [n_samples, n_features, n_neurons]
            from_behind = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X,federated=federated,
                sample_weights = weights if aggregate else None,
                normalise_sample_weights = normalise_sample_weights)
            if from_behind.ndim == 3 and from_behind.shape[1] == 1:
                # Dense outputs may carry a singleton dim (N,1,units); remove it to avoid N×N broadcast.
                from_behind = np.squeeze(from_behind, axis=1)
            if aggregate and from_behind.ndim == 2:
                # Some backends return per-sample gradients in aggregate mode.
                from_behind = np.mean(from_behind, axis=0)

            #print ("shape", from_front.shape, from_behind.shape)
            #print(from_behind)
            if aggregate:
                FIs = from_front * from_behind
            else:
                # from_front: [n_samples, n_features, n_neurons]
                # from_behind: [n_neurons] (聚合后) 或 [n_samples, n_neurons] (未聚合)
                if from_behind.ndim == 1:
                    FIs = from_front * from_behind[np.newaxis, np.newaxis, :]
                else:
                    # from_behind: [n_samples, n_neurons]
                    FIs = from_front * from_behind[:, np.newaxis, :]
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
                        if aggregate:
                            output = _weighted_mean(output, weights)
                        # else: 保留 [n_samples, ...] 维度
                        from_front.append(output)

            from_front = np.asarray(from_front)

            #from_front.shape: [Channel_out * n_mv_0 * n_mv_1, F1, F2, Channel_in] if aggregate
            #                  [Channel_out * n_mv_0 * n_mv_1, n_samples, F1, F2, Channel_in] if not aggregate
            if aggregate:
                if is_channel_first:
                    from_front = from_front.reshape(
                        (n_output_channel,n_mv_0,n_mv_1,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
                else: # channels_last
                    from_front = from_front.reshape(
                        (n_mv_0,n_mv_1,n_output_channel,kernel_shape[0],kernel_shape[1],int(prev_output.shape[-1])))

                # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1]
                # 	or [F1,F2,Channel_in, n_mv_0, n_mv_1,Channel_out]
                from_front = np.moveaxis(from_front, [0,1,2], [3,4,5])
            else:
                # 保留样本维度的reshape
                n_samples = target_X.shape[0]
                if is_channel_first:
                    from_front = from_front.reshape(
                        (n_output_channel,n_mv_0,n_mv_1,n_samples,kernel_shape[0],kernel_shape[1],int(prev_output.shape[1])))
                    from_front = np.moveaxis(from_front, [0,1,2,3], [4,5,6,0])  # [n_samples, F1, F2, Channel_in, Channel_out, n_mv_0, n_mv_1]
                else: # channels_last
                    from_front = from_front.reshape(
                        (n_mv_0,n_mv_1,n_output_channel,n_samples,kernel_shape[0],kernel_shape[1],int(prev_output.shape[-1])))
                    from_front = np.moveaxis(from_front, [0,1,2,3], [4,5,6,0])  # [n_samples, F1, F2, Channel_in, n_mv_0, n_mv_1, Channel_out]

            # [Channel_out, H_out(n_mv_0), W_out(n_mv_1)] or [n_samples, Channel_out, H_out, W_out]
            from_behind = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X, by_batch = False,federated=federated,
                sample_weights = weights if aggregate else None,
                normalise_sample_weights = normalise_sample_weights)
            if aggregate and from_behind.ndim == 4:
                # Some backends return per-sample gradients even in aggregate mode.
                # Reduce over samples to match the [C_out, H, W] shape expected here.
                from_behind = np.mean(from_behind, axis=0)

            #t1 = time.time()
            # [F1,F2,Channel_in, Channel_out, n_mv_0, n_mv_1] (channels_firs)
            # or [F1,F2,Channel_in,n_mv_0, n_mv_1,Channel_out] (channels_last)
            #print(from_front.shape,from_behind.shape)
            if aggregate:
                FIs = from_front * from_behind
            else:
                # from_front: [n_samples, F1, F2, Channel_in, ...]
                # from_behind: [Channel_out, H_out, W_out] 或 [n_samples, Channel_out, H_out, W_out]
                if from_behind.ndim == 3:
                    # 需要广播到样本维度
                    FIs = from_front * from_behind[np.newaxis, ...]
                else:
                    # 已经有样本维度
                    FIs = from_front * from_behind[:, np.newaxis, np.newaxis, np.newaxis, ...]


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
            if aggregate:
                if is_channel_first:
                    FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [F1, F2, Channel_in, Channel_out]
                else:
                    if not federated:
                        np.save("origAct",masked_front)
                        np.save("origOut",masked_behind)
                    FIs = np.sum(np.sum(FIs, axis=-2), axis=-2)
            else:
                # 非聚合模式：需要保留样本维度
                if is_channel_first:
                    # [n_samples, F1, F2, Channel_in, Channel_out, n_mv_0, n_mv_1]
                    FIs = np.sum(np.sum(FIs, axis = -1), axis = -1) # [n_samples, F1, F2, Channel_in, Channel_out]
                else:
                    # [n_samples, F1, F2, Channel_in, n_mv_0, n_mv_1, Channel_out]
                    FIs = np.sum(np.sum(FIs, axis=-3), axis=-3) # [n_samples, F1, F2, Channel_in, Channel_out]


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
        elif model_util.is_BatchNorm(lname):
            # FI = front (activation influence) * back (output gradient), per gamma/beta channel
            # Get BN output activations
            if idx_to_tl == 0 or idx_to_tl - 1 == 0:
                bn_out = model.layers[idx_to_tl](target_X, training=False)
                bn_out = K.get_session().run(bn_out)
            else:
                t_model = Model(inputs=model.input, outputs=model.layers[idx_to_tl].output)
                bn_out = t_model.predict(target_X)

            layer_config = model.layers[idx_to_tl].get_config()
            axis = layer_config.get('axis', -1)
            # move channel axis to last for consistent reduction
            if isinstance(axis, (list, tuple)):
                axis = axis[0]
            if axis < 0:
                axis = bn_out.ndim + axis
            if axis != bn_out.ndim - 1:
                bn_out = np.moveaxis(bn_out, axis, -1)

            def _reduce_to_channels(arr):
                arr = np.abs(arr)
                if arr.ndim <= 1:
                    return arr
                reshaped = arr.reshape(arr.shape[0], -1, arr.shape[-1])
                return np.mean(reshaped, axis=1)

            front_base = _reduce_to_channels(bn_out)

            # NOTE: normalization disabled for inspection
            # front_gamma = norm_scaler.fit_transform(front_base)
            # front_beta = norm_scaler.fit_transform(np.ones_like(front_base))
            front_gamma = front_base
            front_beta = np.ones_like(front_base)
            if aggregate:
                front_gamma = _weighted_mean(front_gamma, weights)
                front_beta = _weighted_mean(front_beta, weights)

            back_full = compute_gradient_to_output(
                path_to_keras_model, idx_to_tl, target_X, by_batch=False,
                federated=federated, sample_weights=weights if aggregate else None,
                normalise_sample_weights=normalise_sample_weights)

            if aggregate:
                if back_full.ndim > 1:
                    back_ch = np.mean(back_full.reshape(-1, back_full.shape[-1]), axis=0)
                else:
                    back_ch = back_full

                FIs_gamma = front_gamma * back_ch
                FIs_beta = front_beta * back_ch
                FIs = np.asarray([FIs_gamma, FIs_beta])

                masked_front = np.asarray([front_gamma, front_beta])
                masked_behind = np.asarray([back_ch, back_ch])
            else:
                # 保留样本维度
                if back_full.ndim > 1:
                    back_ch = np.mean(back_full.reshape(back_full.shape[0], -1, back_full.shape[-1]), axis=1)
                else:
                    back_ch = back_full

                FIs_gamma = front_gamma * back_ch  # [n_samples, n_channels]
                FIs_beta = front_beta * back_ch    # [n_samples, n_channels]
                FIs = np.asarray([FIs_gamma, FIs_beta])  # [2, n_samples, n_channels]
                FIs = np.transpose(FIs, (1, 0, 2))  # [n_samples, 2, n_channels]

                masked_front = np.asarray([front_gamma, front_beta])
                masked_behind = np.asarray([back_ch, back_ch])

            grad_scndcr = None
            if use_gradient_loss:
                grad_scndcr = compute_gradient_to_loss(
                    path_to_keras_model, idx_to_tl, target_X, target_y,
                    by_batch=False, loss_func=loss_func, federated=federated)
                if isinstance(grad_scndcr, list):
                    grad_scndcr = np.asarray(grad_scndcr[:len(t_w)])
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
                path_to_keras_model, idx_to_tl, target_X, by_batch = True, sample_weights = weights,
                normalise_sample_weights = normalise_sample_weights) # shape = (num_units,)

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

            if aggregate:
                # 原来的聚合逻辑
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
            else:
                # 新逻辑：保留样本维度
                # FIs: [n_samples, n_features, n_neurons] or similar
                # 需要重塑为 [n_samples, n_params]
                original_shape = FIs.shape[1:]  # 保存权重形状（不含样本维度）
                FIs_flat = FIs.reshape(FIs.shape[0], -1)  # [n_samples, n_params]
                total_cands[idx_to_tl] = {
                    'shape': original_shape,
                    'costs': FIs_flat,
                    'per_sample': True
                }
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


def compute_FI_and_GL_by_sample(
    X, y,
    indices_to_target,
    target_weights,
    is_multi_label = True,
    path_to_keras_model = None,
    federated = False,
    sample_weights = None,  # kept for API compatibility but not used
    normalise_sample_weights = True,
    use_gradient_loss = True):
    """
    Per-sample version of compute_FI_and_GL.
    Uses aggregate=False to compute all samples in one batch while preserving
    the sample dimension, which is much more efficient than repeated single-sample
    evaluations.

    Note: sample_weights parameter is kept for API compatibility but is not used
    in non-aggregated mode to preserve per-sample FI values.
    """
    return compute_FI_and_GL(
        X, y, indices_to_target, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=federated,
        sample_weights=None,  # 不使用样本权重，保持所有样本同等权重
        normalise_sample_weights=normalise_sample_weights,
        use_gradient_loss=use_gradient_loss,
        aggregate=False)


# LSTM：计算单个权重的前向影响
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
        # NOTE: normalization disabled for inspection
        # out = norm_scaler.fit_transform(out.flatten().reshape(1,-1)).reshape(-1,)
        out = out.reshape(original_shape)

    # N = num_features or num_features + num_units
    out = np.einsum('ijk,ij->ijk', out, const) # shape = (batch_size, time_steps, N)
    return out


# LSTM：为门控计算常量项
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


# LSTM：计算前向影响（所有门控/权重）
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
        # NOTE: normalization disabled for inspection
        # scaled_out_combined = norm_scaler.fit_transform(np.abs(out_combined).flatten().reshape(1,-1))
        # scaled_out_combined = scaled_out_combined.reshape(original_shape)
        scaled_out_combined = np.abs(out_combined)
        # mean out_combined's shape: ((num_features + num_units) * 4,)
        # for each neural weight, the average over both time step and the batch
        avg_scaled_out_combined = np.mean(
            scaled_out_combined.reshape(-1, scaled_out_combined.shape[-1]), axis = 0)
        from_front.append(avg_scaled_out_combined)

    # from_front's shape = (num_units, (num_features + num_units) * 4)
    from_front = np.asarray(from_front)
    print ("For lstm's front part of FI: {}".format(from_front.shape))
    return from_front, gate_orders


# BL 方法：changed/unchanged 分别计算并做 Pareto 过滤
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


def _select_by_confidence(indices, confidences, sample_size, top=True):
    """
    Choose sample_size entries from indices according to confidence ordering.
    top=True => highest confidence; top=False => lowest confidence.
    """
    if sample_size <= 0 or len(indices) == 0:
        return np.zeros(0, dtype=int)
    idx_arr = np.asarray(indices)
    conf_arr = np.asarray(confidences)
    order = np.argsort(conf_arr)
    if top:
        sel = order[::-1][:sample_size]
    else:
        sel = order[:sample_size]
    return idx_arr[sel]


def sample_for_extent_sbfl(
    idx_pass, idx_fail,
    probs, pred_labels, true_labels,
    sample_size=None):
    """
    Sample based on error extent.

    Sampling:
    - Fail: select samples with highest error degree (most confidently wrong)
    - Pass: select samples with highest correctness (most confidently right)

    Returns:
        selected_idx_pass, selected_idx_fail, pass_weights, fail_weights
    """
    idx_pass = np.asarray(idx_pass, dtype=int)
    idx_fail = np.asarray(idx_fail, dtype=int)
    if sample_size is None:
        sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size <= 0 or (len(idx_pass) == 0 and len(idx_fail) == 0):
        return (np.array([], dtype=int), np.array([], dtype=int),
                np.array([], dtype=float), np.array([], dtype=float))

    probs_arr = np.asarray(probs)
    pred_labels = np.asarray(pred_labels)
    true_labels = np.asarray(true_labels)

    def compute_error_degree(indices):
        if len(indices) == 0:
            return np.array([], dtype=float)
        if probs_arr.ndim == 2 and probs_arr.shape[1] > 1:
            true_probs = probs_arr[indices, true_labels[indices]]
            pred_probs = probs_arr[indices, pred_labels[indices]]
            error_degree = pred_probs - true_probs
        else:
            probs_flat = probs_arr.flatten() if probs_arr.ndim > 1 else probs_arr
            error_degree = np.abs(probs_flat[indices] - true_labels[indices])
        return np.clip(error_degree, 0.0, 1.0)

    def compute_correctness(indices):
        if len(indices) == 0:
            return np.array([], dtype=float)
        if probs_arr.ndim == 2 and probs_arr.shape[1] > 1:
            correctness = probs_arr[indices, true_labels[indices]]
        else:
            probs_flat = probs_arr.flatten() if probs_arr.ndim > 1 else probs_arr
            correctness = np.where(
                true_labels[indices] == 1,
                probs_flat[indices],
                1.0 - probs_flat[indices]
            )
        return np.clip(correctness, 0.0, 1.0)

    all_fail_error = compute_error_degree(idx_fail)
    all_pass_correct = compute_correctness(idx_pass)

    if len(idx_fail) > 0:
        if len(idx_fail) > sample_size:
            fail_order = np.argsort(all_fail_error)[::-1]
            selected_fail_positions = fail_order[:sample_size]
            selected_idx_fail = idx_fail[selected_fail_positions]
            fail_weights = all_fail_error[selected_fail_positions]
        else:
            selected_idx_fail = idx_fail
            fail_weights = all_fail_error
    else:
        selected_idx_fail = np.array([], dtype=int)
        fail_weights = np.array([], dtype=float)

    if len(idx_pass) > 0:
        if len(idx_pass) > sample_size:
            pass_order = np.argsort(all_pass_correct)[::-1]
            selected_pass_positions = pass_order[:sample_size]
            selected_idx_pass = idx_pass[selected_pass_positions]
            pass_weights = all_pass_correct[selected_pass_positions]
        else:
            selected_idx_pass = idx_pass
            pass_weights = all_pass_correct
    else:
        selected_idx_pass = np.array([], dtype=int)
        pass_weights = np.array([], dtype=float)

    return selected_idx_pass, selected_idx_fail, pass_weights, fail_weights


# GL 方法：changed/unchanged 的损失梯度比值
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


# 随机定位（对照基线）
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

# 缓存并获取指定层的输出
def _get_layer_output(model, layer_idx, X):
    from tensorflow.keras.models import Model
    if not hasattr(_get_layer_output, "_cache"):
        _get_layer_output._cache = {}
    cache = _get_layer_output._cache
    if layer_idx not in cache:
        cache[layer_idx] = Model(inputs=model.input, outputs=model.layers[layer_idx].output)
    return cache[layer_idx].predict(X)


# 计算层级激活覆盖（按样本权重）
def _coverage_per_unit(
    model, layer_idx, X, sample_weights, activation_threshold=0.1, normalise_sample_weights=True):
    """
    Return coverage per output unit/channel for a given layer, weighted by sample_weights.
    For Dense: shape (out_units,)
    For Conv2D: shape (out_channels,)
    When normalise_sample_weights is False, this returns weighted sums (not averages).
    """
    if X is None or len(X) == 0:
        return None
    out = _get_layer_output(model, layer_idx, X)
    act = np.abs(out)
    # reduce spatial dims for conv
    layer = model.layers[layer_idx]
    lname = type(layer).__name__
    if model_util.is_FC(lname):
        # (num_samples, out_units)
        act_unit = act
    elif model_util.is_C2D(lname):
        data_format = getattr(layer, "data_format", "channels_last")
        if data_format == "channels_first":
            act_unit = np.max(act, axis=(2, 3))  # (num_samples, out_channels)
        else:
            act_unit = np.max(act, axis=(1, 2))  # (num_samples, out_channels)
    else:
        return None
    max_abs = np.max(act_unit) if act_unit.size else 0.0
    if max_abs > 0:
        act_unit = act_unit / max_abs
    cov_unit = (act_unit >= activation_threshold).astype(float)
    if normalise_sample_weights:
        weights = _normalise_sample_weights(cov_unit.shape[0], sample_weights)
    else:
        weights = _clean_sample_weights(cov_unit.shape[0], sample_weights)
    if weights is None:
        return np.mean(cov_unit, axis=0)
    return np.tensordot(weights, cov_unit, axes=([0], [0]))


# 基于覆盖的可疑度计算（Ochiai 形式）
def _suspicious_from_cov(fail_cov, pass_cov, target_shape, eps=1e-12):
    fail_vals = np.repeat(fail_cov, int(np.prod(target_shape[:-1]))) if len(target_shape) > 2 else fail_cov
    pass_vals = np.repeat(pass_cov, int(np.prod(target_shape[:-1]))) if len(target_shape) > 2 else pass_cov
    total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
    sus = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
    return sus


# 把覆盖可疑度展开到权重索引上
def _build_results_from_cov(layer_idx, lname, t_w, fail_cov, pass_cov, total_failed, eps=1e-12):
    costs_and_keys = []
    indices_to_nodes = []
    total_failed = float(total_failed) if total_failed is not None else 0.0
    total_failed = total_failed if total_failed > 0 else eps
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
    suspiciousness = fail_vals / np.sqrt(total_failed * (fail_vals + pass_vals) + eps)
    for i, sus in enumerate(suspiciousness):
        costs_and_keys.append(([layer_idx, i], sus))
        indices_to_nodes.append([layer_idx, np.unravel_index(i, target_shape)])
    return costs_and_keys


# 从 cost 结构中提取 FI 向量
def _extract_costs(entry):
    costs = entry.get("costs", None)
    if costs is None:
        return None
    arr = np.asarray(costs)
    if arr.ndim > 1 and arr.shape[1] >= 2:
        # assume last column corresponds to FI
        return arr[:, -1]
    return arr.reshape(-1)


# 从 cost 列表中提取 FI 向量列表
def _extract_costs_list(entry_list):
    res = []
    for c in entry_list:
        arr = np.asarray(c)
        if arr.ndim > 1 and arr.shape[1] >= 2:
            res.append(arr[:, -1])
        else:
            res.append(arr.reshape(-1))
    return res


def _ensure_probabilities(predictions, eps=1e-7):
    """
    将模型输出统一转换成“概率”形式（用于后续 confidence 计算）。
    处理逻辑：
    - 已经是概率（范围在 [0,1] 且每行求和≈1）时，直接返回。
    - 否则视为 logits：二分类用 sigmoid，多分类用 softmax。
    - 兼容形状：(N, 1, C) / (N,) / (N,1) / (N,C)。
    """
    # 转成 numpy，方便统一处理
    preds = np.asarray(predictions)
    # 有些模型输出形状是 (N, 1, C)，这里把中间的 1 维 squeeze 掉
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    # 二分类：一维向量（N,）
    if preds.ndim == 1:
        # 如果数值超出 [0,1]，说明是 logits，转成 sigmoid 概率
        if np.any(preds < 0) or np.any(preds > 1):
            return 1.0 / (1.0 + np.exp(-preds))
        # 否则已经是概率
        return preds
    # 形状 (N,1) 的二分类输出，压成一维再递归处理
    if preds.ndim == 2 and preds.shape[1] == 1:
        return _ensure_probabilities(preds.reshape(-1), eps)
    # 多分类：先判断是否已是概率分布
    if np.all(preds >= -eps) and np.all(preds <= 1.0 + eps):
        row_sums = preds.sum(axis=1)
        if np.allclose(row_sums, 1.0, atol=1e-3):
            return preds
    # 若不是概率，则视为 logits，做数值稳定的 softmax
    maxes = np.max(preds, axis=1, keepdims=True)  # 减最大值避免溢出
    exp = np.exp(preds - maxes)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _temperature_scale_probs(probs, temperature=1.0, eps=1e-7):
    """
    Apply temperature scaling to probabilities.
    For binary: scale in logit space; for multi-class: softmax with temperature via log-prob.
    """
    if temperature is None or temperature == 1.0:
        return probs
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    if p.ndim == 1:
        logit = np.log(p) - np.log(1.0 - p)
        return 1.0 / (1.0 + np.exp(-logit / temperature))
    if p.ndim == 2 and p.shape[1] == 1:
        return _temperature_scale_probs(p.reshape(-1), temperature, eps).reshape(-1, 1)
    logp = np.log(p)
    scaled = np.exp(logp / temperature)
    return scaled / np.sum(scaled, axis=1, keepdims=True)


def _margin_weights(probs, true_labels, pred_labels, gamma=1.0):
    """
    Compute per-sample margin weights from probabilities.
    - pass: p_true - p_top2
    - fail: p_top1 - p_true
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = np.stack([1.0 - p, p], axis=1)
    top2 = np.partition(p, -2, axis=1)[:, -2]
    idx = np.arange(len(pred_labels))
    p_true = p[idx, true_labels]
    p_pred = p[idx, pred_labels]
    margin = np.where(pred_labels == true_labels, p_true - top2, p_pred - p_true)
    margin = np.clip(margin, a_min=0.0, a_max=None)
    if gamma is not None and gamma != 1.0:
        margin = margin ** gamma
    return margin


def _sparsify_topk(vals, topk_frac):
    """
    Keep only the top-k fraction of values (by magnitude) and zero out the rest.
    """
    if vals is None or topk_frac is None or topk_frac >= 1.0:
        return vals
    flat = np.asarray(vals, dtype=float).reshape(-1)
    if flat.size == 0:
        return vals
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    k = max(1, int(np.ceil(topk_frac * flat.size)))
    thresh = np.partition(flat, -k)[-k]
    mask = np.asarray(vals) >= thresh
    return np.where(mask, vals, 0.0)


def _normalise_fi_vector(vals, eps=1e-12):
    """
    Normalise FI scores so that the total magnitude sums to 1, avoiding division by zero.
    """
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return arr
    total = np.sum(np.abs(arr))
    if total <= eps:
        return arr
    return arr / total


# SBFL 变体：FI 参与度 + 置信度结果（新版实现，覆盖上面同名函数）
def localise_by_qexec_qres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, return_details=False):
    """
    Quantised execution (FI-based) + quantised result (probability) SBFL (Ochiai).
    Confidence is used purely as a result signal; FI remains unweighted.
    (与传统 SBFL 对齐：执行度与结果信号在外层结合，而不是在 FI 计算时加权。)
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
        pred_confidence = np.where(pred_labels == 1, probs, 1.0 - probs)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
        pred_confidence = probs[np.arange(len(pred_labels)), pred_labels]
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y

    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        pass_conf = pred_confidence[idx_pass]
        fail_conf = pred_confidence[idx_fail]
        idx_pass = _select_by_confidence(idx_pass, pass_conf, sample_size, top=True)
        idx_fail = _select_by_confidence(idx_fail, fail_conf, sample_size, top=False)

    pass_conf = pred_confidence[idx_pass] if len(idx_pass) else np.array([])
    fail_conf = pred_confidence[idx_fail] if len(idx_fail) else np.array([])

    fail_conf_w = np.clip(fail_conf, 0.0, 1.0)
    pass_conf_w = np.clip(pass_conf, 0.0, 1.0)
    fail_conf_sum = float(np.sum(fail_conf_w)) if len(fail_conf_w) else 0.0
    fail_conf_sum = fail_conf_sum if fail_conf_sum > 0 else eps

    # Batch-aggregated FI (no per-sample matrix) for fail/pass.
    fail_cands = compute_FI_and_GL(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=fail_conf_w if len(idx_fail) else None,
        normalise_sample_weights=False,
        use_gradient_loss=False,
        aggregate=True)
    pass_cands = compute_FI_and_GL(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=pass_conf_w if len(idx_pass) else None,
        normalise_sample_weights=False,
        use_gradient_loss=False,
        aggregate=True)

    costs_and_keys = []
    details = []  # optional: keep raw fail/pass FI (after confidence scaling)

    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape")
            if target_shape is None:
                target_shape = np.asarray(t_w).shape
            n_params = int(np.prod(target_shape))
            fail_vals = _extract_costs(fail_entry)
            pass_vals = _extract_costs(pass_entry)
            if fail_vals is None:
                fail_vals = np.zeros(n_params)
            if pass_vals is None:
                pass_vals = np.zeros(n_params)
            if fail_vals.shape != pass_vals.shape:
                if fail_vals.shape[0] == 0:
                    fail_vals = np.zeros_like(pass_vals)
                elif pass_vals.shape[0] == 0:
                    pass_vals = np.zeros_like(fail_vals)
            fail_sum = _normalise_fi_values(fail_vals)
            pass_sum = _normalise_fi_values(pass_vals)

            suspiciousness = fail_sum / np.sqrt(fail_conf_sum * (fail_sum + pass_sum) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
                if return_details:
                    details.append([idx_to_tl, i, sus, fail_sum[i], pass_sum[i]])
        else:
            # Fallback for LSTM: use aggregated costs when per-sample matrices are unavailable.
            shapes = fail_entry.get("shape") or pass_entry.get("shape") or [w.shape for w in t_w]
            fail_list = fail_entry.get("costs", [])
            pass_list = pass_entry.get("costs", [])
            fail_vals_list = _extract_costs_list(fail_list) if fail_list else []
            pass_vals_list = _extract_costs_list(pass_list) if pass_list else []
            for idx_to_w, shape in enumerate(shapes):
                fail_vals = fail_vals_list[idx_to_w] if idx_to_w < len(fail_vals_list) else np.zeros(np.prod(shape))
                pass_vals = pass_vals_list[idx_to_w] if idx_to_w < len(pass_vals_list) else np.zeros(np.prod(shape))
                fail_vals = _normalise_fi_values(fail_vals)
                pass_vals = _normalise_fi_values(pass_vals)
                fail_sum = fail_vals
                pass_sum = pass_vals
                suspiciousness = fail_sum / np.sqrt(fail_conf_sum * (fail_sum + pass_sum) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))
                    if return_details:
                        details.append([(idx_to_tl, idx_to_w), local_i, sus, fail_sum[local_i], pass_sum[local_i]])

    sorted_costs_and_keys = sorted(costs_and_keys, key=lambda v: v[1], reverse=True)
    if return_details:
        import pandas as pd
        det_df = pd.DataFrame(details, columns=["layer", "flat_idx", "suspiciousness", "fail_fi", "pass_fi"])
        return sorted_costs_and_keys, det_df
    return sorted_costs_and_keys


def localise_by_qexec_qres_error_extent_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, return_details=False):
    """
    Quantised execution + quantised result with error extent sampling and weighting.

    Sampling: Based on error extent
    - Fail: select samples with highest error degree (most confidently wrong)
    - Pass: select samples with highest correctness (most confidently right)

    Weighting: Error extent
    - Error degree = predicted_class_prob - true_class_prob
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y

    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]

    # Use error-extent sampling
    idx_pass, idx_fail, pass_weights, fail_weights = sample_for_extent_sbfl(
        idx_pass, idx_fail,
        probs, pred_labels, true_labels,
        sample_size=None)

    if len(fail_weights) > 0:
        print(
            f"[qres_error_extent] Fail: n={len(idx_fail)}, "
            f"error_degree: min={fail_weights.min():.4f}, max={fail_weights.max():.4f}, "
            f"mean={fail_weights.mean():.4f}, std={fail_weights.std():.4f}"
        )
    if len(pass_weights) > 0:
        print(
            f"[qres_error_extent] Pass: n={len(idx_pass)}, "
            f"correctness: min={pass_weights.min():.4f}, max={pass_weights.max():.4f}, "
            f"mean={pass_weights.mean():.4f}"
        )

    # Per-sample FI (matrix) for fail/pass to support extent-weighted Ochiai.
    fail_cands = compute_FI_and_GL_by_sample(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL_by_sample(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)

    fail_error_w = fail_weights
    pass_correct_w = pass_weights
    fail_error_sum = float(np.sum(fail_error_w)) if len(fail_error_w) else 0.0
    fail_error_sum = fail_error_sum if fail_error_sum > 0 else eps

    costs_and_keys = []
    details = []

    def _sum_weighted(conf_w, mat, n_params):
        if mat.size == 0:
            return np.zeros(n_params)
        if mat.ndim == 1:
            vec = mat.reshape(-1)
            if vec.size != n_params:
                vec = vec[:n_params]
            return _normalise_fi_values(vec)
        if mat.shape[0] != len(conf_w):
            min_len = min(mat.shape[0], len(conf_w))
            mat = mat[:min_len]
            conf_w = conf_w[:min_len]
        mat = _normalise_fi_values(mat)
        vec = conf_w @ mat
        if vec.size != n_params:
            vec = vec[:n_params]
        return vec

    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape")
            if target_shape is None:
                target_shape = np.asarray(t_w).shape
            n_params = int(np.prod(target_shape))
            fail_mat = np.asarray(fail_entry.get("costs", []))
            pass_mat = np.asarray(pass_entry.get("costs", []))

            fail_sum = _sum_weighted(fail_error_w, fail_mat, n_params)
            pass_sum = _sum_weighted(pass_correct_w, pass_mat, n_params)

            suspiciousness = fail_sum / np.sqrt(fail_error_sum * (fail_sum + pass_sum) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
                if return_details:
                    details.append([idx_to_tl, i, sus, fail_sum[i], pass_sum[i]])
        else:
            shapes = fail_entry.get("shape") or pass_entry.get("shape") or [w.shape for w in t_w]
            fail_list = fail_entry.get("costs", [])
            pass_list = pass_entry.get("costs", [])
            fail_vals_list = _extract_costs_list(fail_list) if fail_list else []
            pass_vals_list = _extract_costs_list(pass_list) if pass_list else []
            pass_correct_sum = float(np.sum(pass_correct_w)) if len(pass_correct_w) else 0.0
            for idx_to_w, shape in enumerate(shapes):
                fail_vals = fail_vals_list[idx_to_w] if idx_to_w < len(fail_vals_list) else np.zeros(np.prod(shape))
                pass_vals = pass_vals_list[idx_to_w] if idx_to_w < len(pass_vals_list) else np.zeros(np.prod(shape))
                fail_vals = _normalise_fi_values(fail_vals)
                pass_vals = _normalise_fi_values(pass_vals)
                fail_sum = fail_vals * fail_error_sum
                pass_sum = pass_vals * pass_correct_sum
                suspiciousness = fail_sum / np.sqrt(fail_error_sum * (fail_sum + pass_sum) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))
                    if return_details:
                        details.append([(idx_to_tl, idx_to_w), local_i, sus, fail_sum[local_i], pass_sum[local_i]])

    sorted_costs_and_keys = sorted(costs_and_keys, key=lambda v: v[1], reverse=True)
    if return_details:
        import pandas as pd
        det_df = pd.DataFrame(details, columns=["layer", "flat_idx", "suspiciousness", "fail_fi", "pass_fi"])
        return sorted_costs_and_keys, det_df
    return sorted_costs_and_keys


# SBFL 变体：qexec + qres（B+C 改进版）
def localise_by_qexec_qres_bc_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, temperature=5.0, margin_gamma=1.0, fi_topk_frac=0.1):
    """
    Quantised execution (FI-based) + margin-weighted qres with temperature scaling (B)
    and FI top-k sparsification (C).
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs_raw = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs_raw >= 0.5).astype(int)
    else:
        probs_raw = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs_raw, axis=1)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y

    probs = _temperature_scale_probs(probs_raw, temperature=temperature)
    margin_scores = _margin_weights(probs, true_labels, pred_labels, gamma=margin_gamma)

    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        rng = np.random.default_rng(sample_seed)
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)

    fail_cands = compute_FI_and_GL(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=margin_scores[idx_fail] if len(idx_fail) else None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=margin_scores[idx_pass] if len(idx_pass) else None,
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
            fail_vals = _normalise_fi_values(fail_vals)
            pass_vals = _normalise_fi_values(pass_vals)
            fail_vals = _sparsify_topk(fail_vals, fi_topk_frac)
            pass_vals = _sparsify_topk(pass_vals, fi_topk_frac)
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
                fail_vals = _normalise_fi_values(fail_vals)
                pass_vals = _normalise_fi_values(pass_vals)
                fail_vals = _sparsify_topk(fail_vals, fi_topk_frac)
                pass_vals = _sparsify_topk(pass_vals, fi_topk_frac)
                total_fail_weight = float(np.sum(fail_vals)) if np.sum(fail_vals) > 0 else eps
                suspiciousness = fail_vals / np.sqrt(total_fail_weight * (fail_vals + pass_vals) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)


# SBFL 变体：FI 参与度 + 二值结果（按置信度加权）
def localise_by_qexec_bres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, return_details=False):
    """
    Quantised execution (FI-based) + binary result (pass/fail=1).
    FI is unweighted; result weights are constant (1.0) for both pass/fail.
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
        pred_confidence = np.where(pred_labels == 1, probs, 1.0 - probs)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
        pred_confidence = probs[np.arange(len(pred_labels)), pred_labels]
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y
    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        pass_conf = pred_confidence[idx_pass]
        fail_conf = pred_confidence[idx_fail]
        idx_pass = _select_by_confidence(idx_pass, pass_conf, sample_size, top=True)
        idx_fail = _select_by_confidence(idx_fail, fail_conf, sample_size, top=False)

    # Per-sample FI (matrix) for fail/pass; result weights are binary (1.0 per sample).
    fail_cands = compute_FI_and_GL_by_sample(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL_by_sample(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)

    fail_weights = np.ones(len(idx_fail), dtype=float)
    pass_weights = np.ones(len(idx_pass), dtype=float)
    fail_weight_sum = float(np.sum(fail_weights)) if len(fail_weights) else 0.0
    fail_weight_sum = fail_weight_sum if fail_weight_sum > 0 else eps
    pass_weight_sum = float(np.sum(pass_weights)) if len(pass_weights) else 0.0

    costs_and_keys = []
    details = []

    def _sum_weighted(weights, mat, n_params):
        if mat.size == 0:
            return np.zeros(n_params)
        if mat.ndim == 1:
            vec = mat.reshape(-1)
            if vec.size != n_params:
                vec = vec[:n_params]
            return _normalise_fi_values(vec)
        if mat.shape[0] != len(weights):
            min_len = min(mat.shape[0], len(weights))
            mat = mat[:min_len]
            weights = weights[:min_len]
        mat = _normalise_fi_values(mat)
        vec = weights @ mat
        if vec.size != n_params:
            vec = vec[:n_params]
        return vec

    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape")
            if target_shape is None:
                target_shape = np.asarray(t_w).shape
            n_params = int(np.prod(target_shape))
            fail_mat = np.asarray(fail_entry.get("costs", []))
            pass_mat = np.asarray(pass_entry.get("costs", []))

            fail_sum = _sum_weighted(fail_weights, fail_mat, n_params)
            pass_sum = _sum_weighted(pass_weights, pass_mat, n_params)

            suspiciousness = fail_sum / np.sqrt(fail_weight_sum * (fail_sum + pass_sum) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
                if return_details:
                    details.append([idx_to_tl, i, sus, fail_sum[i], pass_sum[i]])
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
                fail_vals = _normalise_fi_values(fail_vals)
                pass_vals = _normalise_fi_values(pass_vals)
                fail_sum = fail_vals * fail_weight_sum
                pass_sum = pass_vals * pass_weight_sum
                suspiciousness = fail_sum / np.sqrt(fail_weight_sum * (fail_sum + pass_sum) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))
                    if return_details:
                        details.append([(idx_to_tl, idx_to_w), local_i, sus, fail_sum[local_i], pass_sum[local_i]])

    sorted_costs_and_keys = sorted(costs_and_keys, key=lambda v: v[1], reverse=True)
    if return_details:
        import pandas as pd
        det_df = pd.DataFrame(details, columns=["layer", "flat_idx", "suspiciousness", "fail_fi", "pass_fi"])
        return sorted_costs_and_keys, det_df
    return sorted_costs_and_keys


def localise_by_qexec_bres_error_extent_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, return_details=False):
    """
    Quantised execution + binary result with error extent sampling.

    Sampling: Based on error extent
    - Fail: select samples with highest error degree (most confidently wrong)
    - Pass: select samples with highest correctness (most confidently right)
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y
    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]

    # Use error-extent sampling
    idx_pass, idx_fail, pass_extent, fail_extent = sample_for_extent_sbfl(
        idx_pass, idx_fail,
        probs, pred_labels, true_labels,
        sample_size=None)

    if len(fail_extent) > 0:
        print(
            f"[bres_error_extent] Fail: n={len(idx_fail)}, "
            f"error_degree: min={fail_extent.min():.4f}, max={fail_extent.max():.4f}, "
            f"mean={fail_extent.mean():.4f}, std={fail_extent.std():.4f}"
        )
    if len(pass_extent) > 0:
        print(
            f"[bres_error_extent] Pass: n={len(idx_pass)}, "
            f"correctness: min={pass_extent.min():.4f}, max={pass_extent.max():.4f}, "
            f"mean={pass_extent.mean():.4f}"
        )

    # Per-sample FI (matrix) for fail/pass; result weights are binary (1.0 per sample).
    fail_cands = compute_FI_and_GL_by_sample(
        X, y, idx_fail, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)
    pass_cands = compute_FI_and_GL_by_sample(
        X, y, idx_pass, target_weights,
        is_multi_label=is_multi_label,
        path_to_keras_model=path_to_keras_model,
        federated=False,
        sample_weights=None,
        use_gradient_loss=False)

    fail_weights = np.ones(len(idx_fail), dtype=float)
    pass_weights = np.ones(len(idx_pass), dtype=float)
    fail_weight_sum = float(np.sum(fail_weights)) if len(fail_weights) else 0.0
    fail_weight_sum = fail_weight_sum if fail_weight_sum > 0 else eps
    pass_weight_sum = float(np.sum(pass_weights)) if len(pass_weights) else 0.0

    costs_and_keys = []
    details = []

    def _sum_weighted(weights, mat, n_params):
        if mat.size == 0:
            return np.zeros(n_params)
        if mat.ndim == 1:
            vec = mat.reshape(-1)
            if vec.size != n_params:
                vec = vec[:n_params]
            return _normalise_fi_values(vec)
        if mat.shape[0] != len(weights):
            min_len = min(mat.shape[0], len(weights))
            mat = mat[:min_len]
            weights = weights[:min_len]
        mat = _normalise_fi_values(mat)
        vec = weights @ mat
        if vec.size != n_params:
            vec = vec[:n_params]
        return vec

    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_entry = fail_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(fail_cands, dict) else {}
        pass_entry = pass_cands.get(idx_to_tl, {"costs": [], "shape": []}) if isinstance(pass_cands, dict) else {}

        if not model_util.is_LSTM(lname):
            target_shape = fail_entry.get("shape") or pass_entry.get("shape")
            if target_shape is None:
                target_shape = np.asarray(t_w).shape
            n_params = int(np.prod(target_shape))
            fail_mat = np.asarray(fail_entry.get("costs", []))
            pass_mat = np.asarray(pass_entry.get("costs", []))

            fail_sum = _sum_weighted(fail_weights, fail_mat, n_params)
            pass_sum = _sum_weighted(pass_weights, pass_mat, n_params)

            suspiciousness = fail_sum / np.sqrt(fail_weight_sum * (fail_sum + pass_sum) + eps)
            for i, sus in enumerate(suspiciousness):
                costs_and_keys.append(([idx_to_tl, i], sus))
                if return_details:
                    details.append([idx_to_tl, i, sus, fail_sum[i], pass_sum[i]])
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
                fail_vals = _normalise_fi_values(fail_vals)
                pass_vals = _normalise_fi_values(pass_vals)
                fail_sum = fail_vals * fail_weight_sum
                pass_sum = pass_vals * pass_weight_sum
                suspiciousness = fail_sum / np.sqrt(fail_weight_sum * (fail_sum + pass_sum) + eps)
                for local_i, sus in enumerate(suspiciousness):
                    costs_and_keys.append(([(idx_to_tl, idx_to_w), local_i], sus))
                    if return_details:
                        details.append([(idx_to_tl, idx_to_w), local_i, sus, fail_sum[local_i], pass_sum[local_i]])

    sorted_costs_and_keys = sorted(costs_and_keys, key=lambda v: v[1], reverse=True)
    if return_details:
        import pandas as pd
        det_df = pd.DataFrame(details, columns=["layer", "flat_idx", "suspiciousness", "fail_fi", "pass_fi"])
        return sorted_costs_and_keys, det_df
    return sorted_costs_and_keys


# SBFL 变体：二值执行覆盖 + 置信度结果
def localise_by_bexec_qres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, normalise_sample_weights=False):
    """
    Binary execution (activation coverage) + quantised result (probability).
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
        pred_confidence = np.where(pred_labels == 1, probs, 1.0 - probs)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
        pred_confidence = probs[np.arange(len(pred_labels)), pred_labels]
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
        rng = np.random.default_rng(sample_seed)
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)
    fail_weights = failure_scores[idx_fail] if len(idx_fail) else None
    pass_weights = success_scores[idx_pass] if len(idx_pass) else None
    total_failed = float(np.sum(fail_weights)) if fail_weights is not None else 0.0

    if "hydra" not in str(path_to_keras_model):
        model = load_model(path_to_keras_model, compile=False)
    else:
        model = load_model_from_h5(path_to_keras_model)

    costs_and_keys = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_cov = _coverage_per_unit(
            model, idx_to_tl, X[idx_fail],
            fail_weights,
            activation_threshold, normalise_sample_weights)
        pass_cov = _coverage_per_unit(
            model, idx_to_tl, X[idx_pass],
            pass_weights,
            activation_threshold, normalise_sample_weights)
        if fail_cov is None and pass_cov is None:
            continue
        if fail_cov is None:
            fail_cov = np.zeros_like(pass_cov)
        if pass_cov is None:
            pass_cov = np.zeros_like(fail_cov)
        costs_and_keys.extend(
            _build_results_from_cov(idx_to_tl, lname, t_w, fail_cov, pass_cov, total_failed, eps)
        )

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)


# SBFL 变体：二值执行覆盖 + 二值结果
def localise_by_bexec_bres_sbfl(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, normalise_sample_weights=False):
    """
    Binary execution (activation coverage) + binary result (pass/fail).
    """
    preds = np.asarray(predictions)
    if preds.ndim == 3 and preds.shape[1] == 1:
        preds = preds[:, 0, :]
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        probs = _ensure_probabilities(preds.reshape(-1))
        pred_labels = (probs >= 0.5).astype(int)
    else:
        probs = _ensure_probabilities(preds)
        pred_labels = np.argmax(probs, axis=1)
    if y.ndim > 1:
        true_labels = np.argmax(y, axis=1)
    else:
        true_labels = y
    correct_mask = pred_labels == true_labels
    idx_pass = np.where(correct_mask)[0]
    idx_fail = np.where(~correct_mask)[0]
    sample_size = min(len(idx_pass), len(idx_fail))
    if sample_size > 0:
        rng = np.random.default_rng(sample_seed)
        idx_pass = rng.choice(idx_pass, sample_size, replace=False)
        idx_fail = rng.choice(idx_fail, sample_size, replace=False)
    fail_weights = np.ones(len(idx_fail), dtype=float) if len(idx_fail) else None
    pass_weights = np.ones(len(idx_pass), dtype=float) if len(idx_pass) else None
    total_failed = float(np.sum(fail_weights)) if fail_weights is not None else 0.0

    if "hydra" not in str(path_to_keras_model):
        model = load_model(path_to_keras_model, compile=False)
    else:
        model = load_model_from_h5(path_to_keras_model)

    costs_and_keys = []
    for idx_to_tl, vs in target_weights.items():
        t_w, lname = vs
        fail_cov = _coverage_per_unit(
            model, idx_to_tl, X[idx_fail],
            fail_weights,
            activation_threshold, normalise_sample_weights)
        pass_cov = _coverage_per_unit(
            model, idx_to_tl, X[idx_pass],
            pass_weights,
            activation_threshold, normalise_sample_weights)
        if fail_cov is None and pass_cov is None:
            continue
        if fail_cov is None:
            fail_cov = np.zeros_like(pass_cov)
        if pass_cov is None:
            pass_cov = np.zeros_like(fail_cov)
        costs_and_keys.extend(
            _build_results_from_cov(idx_to_tl, lname, t_w, fail_cov, pass_cov, total_failed, eps)
        )

    return sorted(costs_and_keys, key=lambda v: v[1], reverse=True)


# Guider：保留旧入口，转调到 bexec_qres_sbfl
def localise_by_bexec_qres_guider(
    X, y, predictions, target_weights,
    path_to_keras_model=None, is_multi_label=True, eps=1e-12, activation_threshold=0.1,
    sample_seed=None, normalise_sample_weights=False):
    """
    Backward-compatible wrapper for bexec_qres_sbfl.
    """
    return localise_by_bexec_qres_sbfl(
        X, y, predictions, target_weights,
        path_to_keras_model=path_to_keras_model,
        is_multi_label=is_multi_label,
        eps=eps,
        activation_threshold=activation_threshold,
        sample_seed=sample_seed,
        normalise_sample_weights=normalise_sample_weights)
