import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import _init_paths as p
from argparse import ArgumentParser
from util import plot_confusion_matrix

def conv_layer(input, features, kernel, strides = 1, padding = 'same', dilation = 1):
	x = tf.keras.layers.Conv1D(features, kernel, strides = strides, padding = padding, dilation_rate = dilation)(input)
	x = tf.keras.layers.experimental.LayerNormalization(scale = False)(x)
	x = tf.keras.layers.PReLU(shared_axes=[1,2])(x)
	return x

def conv_block(input, features, kernels, dilation = 1):
	x = input
	for i in range(len(features)):
		x = conv_layer(x, features[i], kernels[i], dilation = dilation)
		x = conv_layer(x, features[i], kernels[i], strides = 2)

	return x

def get_model(vocab_size):
	in_chars = tf.keras.layers.Input(shape = (None, ))
	x = tf.keras.layers.Embedding(vocab_size + 1, 64)(in_chars)

	x1 = conv_block(x, [128, 128, 128, 128], [9, 7, 5, 3])
	x2 = conv_block(x, [128, 128, 128, 128], [9, 7, 5, 3], dilation = 2)

	x = tf.keras.layers.Concatenate()([x1, x2])
	x = tf.keras.layers.GlobalAveragePooling1D()(x)
	x = tf.keras.layers.Dense(8, activation = 'softmax')(x)

	return tf.keras.models.Model(inputs = in_chars, outputs = x)


def get_compiled_model(vocab_size, lr):
	optim = tf.keras.optimizers.Adam(lr = lr)
	model = get_model(vocab_size)
	model.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

	return model

space = tf.constant(' ')
def _preprocess_feats(*feat_list):
	feat = tf.strings.join([space, *feat_list], separator = ' ')
	feat = tf.strings.substr(feat, 0, 4096)
	return feat

encoding = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
encoder = tfds.features.text.TokenTextEncoder(encoding)
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters = '', lower = False, char_level = True, oov_token = '<unk>')

def encode_label_fn(label):
	return tf.py_function(lambda x: np.asarray(encoder.encode(x.numpy())) - 1, inp=[label], Tout=(tf.int64))

def encode_feat_fn(feat):
	return tf.py_function(lambda x: np.squeeze(np.asarray(tokenizer.texts_to_sequences([x.numpy()]))), inp=[feat], Tout=(tf.int32))

BUFFER_SIZE = 1000
BATCH_SIZE = 32

def load_dataset(dir, features = ['title', 'desc', 'tags'], train = False):
	labels = tf.data.TextLineDataset(os.path.join(dir, 'emotion.txt'))
	labels = labels.map(encode_label_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)

	feat_list = [tf.data.TextLineDataset(os.path.join(dir, f'{feat}.txt')) for feat in features]
	feats = tf.data.Dataset.zip(tuple(feat_list))
	feats = feats.map(_preprocess_feats, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	
	if (train):
		tokenizer.fit_on_texts(tfds.as_numpy(feats))

	feats = feats.map(encode_feat_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	#print(np.histogram([s.size for s in tfds.as_numpy(feats)], bins = 10))

	dataset = tf.data.Dataset.zip((feats, labels))
	if (train):
		dataset = dataset.shuffle(BUFFER_SIZE)
	
	dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
		lambda f, l: tf.size(f),
		[128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 8192, 16384, 32768, 65536],
		[ 32,  32,  32,  32,  32,  32,  32,   32,   32,   32,   32,   32,   32,   32,   32,   32,   16,     8,     4,     2,     1],
		padded_shapes = (tf.TensorShape([None]), tf.TensorShape([1]))
	))

	return dataset

def train(dataset_train, dataset_val, epochs, lr, name):
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath = os.path.join(p.models, f'{name}' + '_{epoch}.h5'),
			save_best_only = False,
			monitor = 'val_loss',
			verbose = 1,
			period = 2
		),
		tf.keras.callbacks.TensorBoard(log_dir = os.path.join(p.logs, 'txt'), histogram_freq = 1),
		tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5,  patience = 4, min_lr = 0.0001, verbose = 1)
	]

	if not os.path.exists(os.path.join(p.logs, 'txt')):
		os.makedirs(os.path.join(p.logs, 'txt'))

	model = get_compiled_model(len(tokenizer.word_index), lr = lr)
	model.fit(dataset_train, epochs = epochs, validation_data = dataset_val, callbacks = callbacks)
	model.save(os.path.join(p.models, f'{name}.h5'))

	return model

def eval(dataset_test, name):
	model = tf.keras.models.load_model(os.path.join(p.models, f'{name}.h5'))
	y_test = dataset_test.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
	y_test = np.fromiter(tfds.as_numpy(y_test), np.int32)
	y_pred = model.predict(dataset_test)
	y_pred = np.argmax(y_pred, axis = 1)
	print(classification_report(y_test, y_pred, target_names = encoding))

	plot_confusion_matrix(y_test, y_pred, encoding, name, save = True)

	return y_test, y_pred

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-n', '--name', default = 'txt')
	parser.add_argument('-f', '--features', nargs = '+', default = ['title', 'desc', 'tags'])
	parser.add_argument('-e', '--epochs', default = 10, type = int)
	parser.add_argument('-lr', '--learning_rate', default = 0.001, type = float)
	parser.add_argument('-t', '--train', action = 'store_true')
	args = parser.parse_args()

	dataset_train = load_dataset(p.train, features = args.features, train = True)
	dataset_val = load_dataset(p.val, features = args.features)

	if (args.train):
		train(dataset_train, dataset_val, args.epochs, args.learning_rate, args.name)

	dataset_test = load_dataset(p.test, features = args.features)
	eval(dataset_test, args.name)
