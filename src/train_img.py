import tensorflow as tf
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import _init_paths as p
from argparse import ArgumentParser
from util import plot_confusion_matrix, _preprocess_image, _preprocess_image_train

def get_model():
	#base_model = tf.keras.applications.DenseNet121(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
	base_model = tf.keras.applications.DenseNet201(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
	for layer in base_model.layers[141:]:
		layer.trainable = False

	x = base_model.output
	x = tf.keras.layers.GlobalMaxPool2D()(x)
	x = tf.keras.layers.Dropout(0.5)(x)
	y = tf.keras.layers.Dense(8, activation = 'softmax')(x)

	return tf.keras.models.Model(inputs = base_model.input, outputs = y)


def get_compiled_model():
	optim = tf.optimizers.SGD(lr = 0.1)
	model = get_model()
	model.compile(optimizer = optim, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

	return model


encoding = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']
encoder = tfds.features.text.TokenTextEncoder(encoding)
def _preprocess_label(label):
	print(label)
	return encoding[label]


def encode_map_fn(label):
	return tf.py_function(lambda x: encoder.encode(x.numpy()), inp=[label], Tout=(tf.int64))


BUFFER_SIZE = 1000
BATCH_SIZE = 32

def load_dataset(dir, train = False):
	labels = tf.data.TextLineDataset(os.path.join(dir, 'emotion.txt'))
	labels = labels.map(encode_map_fn, num_parallel_calls = tf.data.experimental.AUTOTUNE).map(lambda x: x - 1, num_parallel_calls = tf.data.experimental.AUTOTUNE)
	
	preprocess_image = _preprocess_image_train if train else _preprocess_image
	images = tf.data.TextLineDataset(os.path.join(dir, 'images.map'))
	images = images.map(preprocess_image, num_parallel_calls = tf.data.experimental.AUTOTUNE)

	dataset = tf.data.Dataset.zip((images, labels)).cache()
	if (train):
		dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(BUFFER_SIZE, 1))
	
	return dataset.batch(BATCH_SIZE).prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

if not os.path.exists('../training_checkpoints'):
	os.makedirs('../training_checkpoints')
if not os.path.exists('../logs'):
	os.makedirs('../logs')

def train(dataset_train, dataset_val):
	callbacks = [
		tf.keras.callbacks.ModelCheckpoint(
			filepath = '../training_checkpoints/dense_emo_{epoch}.h5',
			save_best_only = False,
			monitor = 'val_loss',
			verbose = 1,
			period = 2
		),
		tf.keras.callbacks.TensorBoard(log_dir = os.path.abspath('../logs'), histogram_freq = 1),
		tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2,  patience = 3, min_lr = 0.0001, verbose = 1)
	]
	model = get_compiled_model()
	model.fit(dataset_train, epochs = 10, validation_data = dataset_val, callbacks = callbacks)
	model.save(os.path.join(p.models, f'img-model.h5'))

	return model

def eval(dataset_test, name):
	model = tf.keras.models.load_model(os.path.join(p.models, f'{name}.h5'))
	y_test = dataset_test.flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices(y))
	y_test = np.fromiter(tfds.as_numpy(y_test), np.int64)
	y_pred = model.predict(dataset_test)
	y_pred = np.argmax(y_pred, axis = 1)
	print(classification_report(y_test, y_pred, target_names = encoding))

	plot_confusion_matrix(y_test, y_pred, encoding, name, save = True)

	return y_test, y_pred


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-n', '--name', default = 'txt')
	parser.add_argument('-f', '--features', nargs = '+', default = ['title', 'desc', 'tags'])
	parser.add_argument('-lr', '--learning_rate', default = 0.001, type = float)
	parser.add_argument('-t', '--train', action = 'store_true')
	args = parser.parse_args()

	dataset_train = load_dataset('../data/train', train = True)
	dataset_val = load_dataset('../data/val')

	if (args.train):
		train(dataset_train, dataset_val)
	
	dataset_test = load_dataset('../data/test')
	eval(dataset_test, args.name)