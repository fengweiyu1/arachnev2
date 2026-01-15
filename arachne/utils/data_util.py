"""
A module for data related functions
"""
import numpy as np

def get_lfw_data(path_to_namedir, is_train = True):
	"""
	"""
	import pickle, os
	if is_train:
		datafile = os.path.join(path_to_namedir, "LFW_train_info.pkl")
	else:
		datafile = os.path.join(path_to_namedir, "LFW_test_info.pkl")
	
	with open(datafile, 'rb') as f:
		data = pickle.load(f)
	
	names = sorted(list(data.keys()))
	at_arr = np.asarray([data[n]['at'] for n in names])
	true_label_arr = np.asarray([data[n]['true'] for n in names]) 
	pred_label_arr = np.asarray([data[n]['pred'] for n in names])
	return {'name':names, 'at':at_arr, 
			'true':true_label_arr, 'pred':pred_label_arr}


def combine(list_of_imgs):
	"""
	"""
	combined = list_of_imgs[0]
	for i in range(1, len(list_of_imgs)):
		combined = np.append(combined, list_of_imgs[i], axis = 0)
	org_shape = combined.shape
	combined = combined.reshape(org_shape[0], 1, org_shape[1], org_shape[2])
	return combined	


def load_rq5_fm_test_val(datadir, which_type = "both"):
	"""
	"""
	import pickle, os
	def get_labels(list_of_imgs):
		labels = []
		for i in range(len(list_of_imgs)):
			for j in range(len(list_of_imgs[i])):
				labels.append(i)
		return np.int32(np.array(labels))

	test_datafile = os.path.join(datadir, 'fmnist_test.pkl')
	val_datafile = os.path.join(datadir, 'fmnist_val.pkl')
	if which_type == 'both':
		with open(test_datafile, 'rb') as f:
			test_X = pickle.load(f)
		test_data = [combine(test_X)/255, get_labels(test_X)]

		with open(val_datafile, 'rb') as f:
			val_X = pickle.load(f)
		val_data =  [combine(val_X)/255, get_labels(val_X)]
		return {'test':test_data, 'val':val_data}
	elif which_type	== 'test':
		with open(test_datafile, 'rb') as f:
			test_X = pickle.load(f)
		test_data = [combine(test_X)/255, get_labels(test_X)]
		return test_data
	elif which_type == 'val':
		with open(val_datafile, 'rb') as f:
			val_X = pickle.load(f)
		val_data =  [combine(val_X)/255, get_labels(val_X)]
		return val_data
	else:
		print ("Wrong type: {}".format(which_type))
		assert False


def load_data(which, path_to_data,
	is_input_2d = False, 
	path_to_female_names = None, 
	path_to_namedir = None):
	import tensorflow as tf
	import os
	
	if which == 'lfw':
		from utils.data_loader import get_LFW_loader
		if path_to_namedir is None:
			path_to_namedir = os.path.dirname(path_to_female_names)
		names_in_train = sorted(get_lfw_data(
			path_to_namedir, is_train = True)['name'])
		names_in_test = sorted(get_lfw_data(
			path_to_namedir, is_train = False)['name'])

		trainloader = get_LFW_loader(image_path = path_to_data,
			split='train', batch_size=1, 
			path_to_female_names = path_to_female_names)
		testloader = get_LFW_loader(image_path = path_to_data,
			split='test', batch_size=1, 
			path_to_female_names = path_to_female_names)

		train_vs = {}
		for image_names, images, image_labels in trainloader:
			train_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
		#
		train_data = [[],[]]
		sorted_train_vs = [train_vs[n] for n in names_in_train]
		train_data[0] = np.array([np.moveaxis(vs[0],[0],[-1]) for vs in sorted_train_vs])
		train_data[1] = np.array([vs[1] for vs in sorted_train_vs])
		
		test_data = [[],[]]
		test_vs = {}
		for image_names, images, image_labels in testloader:
			test_vs[image_names[0]] = [images.numpy()[0], image_labels.item()]
		#
		sorted_test_vs = [test_vs[n] for n in names_in_test]
		test_data[0] = np.array([np.moveaxis(vs[0],[0],[-1]) for vs in sorted_test_vs])
		test_data[1] = np.array([vs[1] for vs in sorted_test_vs])
	elif which in ['fashion_mnist', 'cifar10']:
		if which == 'fashion_mnist':
			# Prefer local raw IDX files under <path_to_data>/fm/FashionMNIST/raw
			import gzip, struct
			raw_dir = os.path.join(path_to_data, "fm", "FashionMNIST", "raw") if isinstance(path_to_data, str) else None

			def read_idx(fname):
				opener = gzip.open if fname.endswith(".gz") else open
				with opener(fname, "rb") as f:
					magic, = struct.unpack(">I", f.read(4))
					ndim = magic & 0xFF
					shape = struct.unpack(">" + "I"*ndim, f.read(4*ndim))
					data = np.frombuffer(f.read(), dtype=np.uint8)
					return data.reshape(shape)

			if raw_dir and os.path.isdir(raw_dir):
				img_train_path = os.path.join(raw_dir, "train-images-idx3-ubyte")
				lbl_train_path = os.path.join(raw_dir, "train-labels-idx1-ubyte")
				img_test_path = os.path.join(raw_dir, "t10k-images-idx3-ubyte")
				lbl_test_path = os.path.join(raw_dir, "t10k-labels-idx1-ubyte")
				# fallback to .gz if plain files not found
				if not os.path.exists(img_train_path) and os.path.exists(img_train_path + ".gz"):
					img_train_path += ".gz"
				if not os.path.exists(lbl_train_path) and os.path.exists(lbl_train_path + ".gz"):
					lbl_train_path += ".gz"
				if not os.path.exists(img_test_path) and os.path.exists(img_test_path + ".gz"):
					img_test_path += ".gz"
				if not os.path.exists(lbl_test_path) and os.path.exists(lbl_test_path + ".gz"):
					lbl_test_path += ".gz"

				train_X = read_idx(img_train_path)
				train_y = read_idx(lbl_train_path)
				test_X = read_idx(img_test_path)
				test_y = read_idx(lbl_test_path)
			else:
				# Fallback to keras loader; some TF versions support 'path', some don't.
				from tensorflow.keras.datasets import fashion_mnist
				try:
					(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data(
						path=path_to_data if isinstance(path_to_data, str) else None)
				except TypeError:
					(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()
			train_X = np.expand_dims(train_X, -1)
			test_X = np.expand_dims(test_X, -1)
			train_data = [train_X.astype(np.float32) / 255.0, train_y.astype(np.int64)]
			test_data = [test_X.astype(np.float32) / 255.0, test_y.astype(np.int64)]
		else:  # cifar10
			# Prefer local CIFAR10 batches under path_to_data; fall back to keras download if not found
			import os, pickle
			def load_cifar_batches(root):
				batch_dir = None
				candidate_dirs = [
					os.path.join(root, "cifar-10-batches-py"),
					os.path.join(root, "train", "cifar-10-batches-py"),
					os.path.join(root, "test", "cifar-10-batches-py"),
				]
				for d in candidate_dirs:
					if os.path.isdir(d) and os.path.isfile(os.path.join(d, "data_batch_1")):
						batch_dir = d
						break
				if batch_dir is None:
					return None

				def load_batch(fname):
					with open(os.path.join(batch_dir, fname), 'rb') as f:
						d = pickle.load(f, encoding='bytes')
						data = d[b'data']
						labels = d[b'labels']
						return data, labels

				train_X_list = []
				train_y_list = []
				for i in range(1,6):
					dat, lbl = load_batch(f"data_batch_{i}")
					train_X_list.append(dat)
					train_y_list.extend(lbl)
				# channels_last for converted model
				train_X = np.vstack(train_X_list).reshape(-1, 3, 32, 32).transpose(0,2,3,1)
				train_y = np.array(train_y_list, dtype=np.int64)
				test_dat, test_lbl = load_batch("test_batch")
				test_X = test_dat.reshape(-1,3,32,32).transpose(0,2,3,1)
				test_y = np.array(test_lbl, dtype=np.int64)
				return (train_X, train_y), (test_X, test_y)

			local = load_cifar_batches(path_to_data)
			if local is None:
				from tensorflow.keras.datasets import cifar10
				(train_X, train_y), (test_X, test_y) = cifar10.load_data()
				train_y = train_y.flatten()
				test_y = test_y.flatten()
				# already channels_last
			else:
				(train_X, train_y), (test_X, test_y) = local

			train_data = [train_X.astype(np.float32) / 255.0, train_y.astype(np.int64)]
			test_data = [test_X.astype(np.float32) / 255.0, test_y.astype(np.int64)]
	elif which in ['GTSRB', 'us_airline']:
		import pickle
		# train
		with open(os.path.join(path_to_data, "train_data.pkl"), 'rb') as f:
			train_data = pickle.load(f)
		if isinstance(train_data, dict):
			train_data = [train_data['data'], train_data['label']]
		
		# test
		with open(os.path.join(path_to_data, "test_data.pkl"), 'rb') as f:
			test_data = pickle.load(f)
		if isinstance(test_data, dict):
			test_data = [test_data['data'], test_data['label']]
		# ensure channels_last for GTSRB
		if which == 'GTSRB':
			if len(train_data[0].shape) == 4 and train_data[0].shape[1] == 3:
				train_data[0] = np.moveaxis(train_data[0], 1, -1)
			if len(test_data[0].shape) == 4 and test_data[0].shape[1] == 3:
				test_data[0] = np.moveaxis(test_data[0], 1, -1)
	else: # for simple_lstm or airline_passengers
		import pickle
		with open(path_to_data, 'rb') as f:
			data = pickle.load(f)
		train_data = data['train']
		test_data = data['test']
		
	return (train_data, test_data)

	
def format_label(labels, num_label):
	"""
	format label which has a integer label to flag type
	e.g., [3, 5] -> [[0,0,1,0,0],[0,0,0,0,1]]
	"""
	num_data = len(labels)
	from collections.abc import Iterable

	new_labels = np.zeros([num_data, num_label])
	for i, v in enumerate(labels):
		if isinstance(v, Iterable):
			new_labels[i,v[0]] = 1
		else:		
			new_labels[i,v] = 1

	return new_labels


def get_indexed_data(indices, datas):
	"""
	return an array of data with indioces
	"""
	new_data = [datas[i] for i in indices]
	return np.asarray(new_data)


def split_into_wrong_and_correct(correct_predictions):
	"""
	Spilt wrong and correct classification result
	Ret (dict):
		ret = {'wrong':indices_to_wrong(list), 
			'correct':indices_to_correct(list)}
	"""
	indices = {'wrong':[], 'correct':[]}
	indices_to_wrong, = np.where(correct_predictions == False)
	indices_to_correct, = np.where(correct_predictions == True)
	indices['correct'] = list(indices_to_correct)
	indices['wrong'] = list(indices_to_wrong)
	return indices


def get_misclf_indices(misclf_indices_file, 
	target_indices = None, 
	use_all = True):
	"""
	"""
	import pandas as pd 

	misclfds = {}
	df = pd.read_csv(misclf_indices_file)
	for i,t,p in zip(df['index'], df['true'], df['pred']):
		if target_indices is not None:
			if i in target_indices:
				key = (t,p)
				if key not in misclfds.keys():
					misclfds[key] = []
				misclfds[key].append(target_indices.index(i))
		else:
			if use_all or i % 2 == 0:
				key = (t,p)
				if key not in misclfds.keys():
					misclfds[key] = []
				misclfds[key].append(i)

	return misclfds


def get_misclf_indices_balanced(df, idx = 0):
	"""
	"""
	misclf_types = list(set(
		[tuple(pair) for pair in df[["true","pred"]].values]))
	ret_misclfds = {}
	for misclf_type in misclf_types:
		misclf_type = tuple(misclf_type)
		true_label, pred_label = misclf_type
		indices_to_misclf = df.loc[
			(df.true == true_label) & (df.pred == pred_label)].index.values
	
		if len(indices_to_misclf) >= 2:
			indices_1, indices_2 = np.array_split(indices_to_misclf, 2)
			ret_misclfds[misclf_type] = indices_1 if idx == 0 else indices_2
		else: # a single input
			ret_misclfds[misclf_type] = indices_to_misclf
	
	return ret_misclfds


def sort_keys_by_cnt(misclfds):
	"""
	"""
	cnts = []
	for misclf_key in misclfds:
		cnts.append([misclf_key, len(misclfds[misclf_key])])
	sorted_keys = [v[0] for v in sorted(cnts, key = lambda v:v[1], reverse = True)]
	return sorted_keys


def get_balanced_dataset(pred_file, top_n, idx = 0):
	"""
	generate the training and test dataset for rq3 ~ rq6 
	idx = 0 or 1 -> to which half, 0 = front half, 1 = latter half
	"""
	import pandas as pd
	from collections.abc import Iterable

	idx = idx if idx == 0 else 1 # only 0 or 1
	target_idx = idx; eval_idx = np.abs(1 - target_idx)
	
	df = pd.read_csv(pred_file, index_col = 'index')
	misclf_df = df.loc[df.true != df.pred]
	misclfds_idx_target = get_misclf_indices_balanced(misclf_df, idx = target_idx)
	sorted_keys = sort_keys_by_cnt(misclfds_idx_target) 
	misclfds_idx_eval = get_misclf_indices_balanced(misclf_df, idx = eval_idx)

	indices_to_corr = df.loc[df.true == df.pred].sort_values(by=['true']).index.values
	indices_to_corr_target = [_idx for i,_idx in enumerate(indices_to_corr) if i % 2 == target_idx]
	indices_to_corr_eval = [_idx for i,_idx in enumerate(indices_to_corr) if i % 2 == eval_idx]

	np.random.seed(0)
	if top_n >= len(sorted_keys):
		msg = "{} is provided when there is only {} number of misclfs".format(
			top_n, len(sorted_keys))
		assert False, msg
	else:
		misclf_key = sorted_keys[top_n]
		misclf_indices = misclfds_idx_target[misclf_key]

		new_data_indices = []; new_test_indices = []
		for sorted_k in sorted_keys: 
			# this means that all incorrect ones are include in new_data
			new_data_indices.extend(misclfds_idx_target[sorted_k])
			new_test_indices.extend(misclfds_idx_eval[sorted_k])
		
		new_data_indices += indices_to_corr_target
		new_test_indices += indices_to_corr_eval
		np.random.shuffle(new_data_indices)
		np.random.shuffle(new_test_indices)	
		return (misclf_key, misclf_indices, new_data_indices, new_test_indices)


def get_misclf_for_rq2(pred_file, percent = 0.1, seed = None):
	"""
	"""
	import pandas as pd
	df = pd.read_csv(pred_file, index_col = 'index')
	misclf_df = df.loc[df.true != df.pred]
	num_to_sample = int(len(misclf_df) * percent)

	np.random.seed(seed)
	indices_to_misclf = np.random.choice(
		misclf_df.index.values, 
		num_to_sample if num_to_sample >= 1 else 1, 
		replace=False)
	
	return indices_to_misclf


def return_chunks(num, batch_size = None):
	if batch_size is None:
		batch_size = num
	num_split = int(np.round(num/batch_size))
	if num_split == 0:
		num_split = 1
	chunks = np.array_split(np.arange(num), num_split)
	return chunks


def get_dataset_for_rq5(pred_file, top_n):
	"""
	"""
	import pandas as pd
	df = pd.read_csv(pred_file, index_col = 'index')
	misclf_df = df.loc[df.true != df.pred]
	misclf_cnts = dict(misclf_df.groupby(['true','pred']).size())
	sorted_misclf_cnts = sorted(
		[vs for vs in misclf_cnts.items()], key = lambda v:v[1], reverse = True)
	top_n_misclf = sorted_misclf_cnts[int(top_n)][0]
 
	indices_to_misclf = df.loc[
		(df.true == top_n_misclf[0]) & (df.pred == top_n_misclf[1])].index.values
	indices_to_corrclf = df.loc[df.true == df.pred].index.values
	return top_n_misclf, indices_to_misclf, indices_to_corrclf
