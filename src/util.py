import _init_paths as p
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import tensorflow as tf

def print_dataset_distribution():
	s = []
	for i, dir in enumerate([p.train, p.val, p.test]):
		with open(os.path.join(dir, 'emotion.txt')) as f:
			s.append(pd.Series([line.strip() for line in f]).value_counts())

	df = pd.concat(s, axis = 1)
	df.loc['Total']= df.sum()
	df['Total'] = df.sum(axis = 1)
	df['%'] = df['Total'] / df.loc['Total', 'Total'] * 100
	print(df)


def plot_confusion_matrix(y_truth, y_pred, encoding, name, save = True):
	cm = confusion_matrix(y_truth, y_pred)
	cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
	cm = pd.DataFrame(cm, index = encoding, columns = encoding).round(2)

	sn.set(font_scale = 1.5)
	plt.figure(figsize = (12, 9))
	ax = sn.heatmap(cm, annot = True, vmin=0, vmax=1, cmap="YlGnBu", square = True)
	ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
	plt.tight_layout()
	if (save):
		plt.savefig(os.path.join(p.root, f'{name}.png'))
	else:
		plt.show()

def _center_crop_square(image):
	shape = tf.shape(image)
	h, w = shape[0], shape[1]
	crop = tf.minimum(h, w)
	y, x = h - crop // 2, w - crop // 2
	return image[y : y + crop , x : x + crop, :]

def _preprocess_image_train(filename):
	img_raw = tf.io.read_file(filename)
	img = tf.io.decode_jpeg(img_raw, channels = 3)
	img = _center_crop_square(img)
	img = tf.image.resize(img, [256, 256])
	img = tf.image.random_crop(img, [224, 224, 3])
	img = tf.image.random_flip_left_right(img)
	img = tf.image.convert_image_dtype(img, tf.float32)
	return img

def _preprocess_image(filename):
	img_raw = tf.io.read_file(filename)
	img = tf.io.decode_jpeg(img_raw, channels = 3)
	img = _center_crop_square(img)
	img = tf.image.resize(img, [224, 224])
	img = tf.image.convert_image_dtype(img, tf.float32)
	return img