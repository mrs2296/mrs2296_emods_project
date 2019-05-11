import glob
import os
import pandas as pd
import flickrapi
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
import json
import time
import requests
import _init_paths as p
import html
import re

def load_urls():
	df = pd.concat([pd.read_csv(f, header=None, names=['emotion', 'url', 'disagree', 'agree']) for f in glob.glob('../data/urls/*[0-9]??_[0-9]*.csv')], ignore_index=True)
	df = df.groupby(['emotion','url'], as_index = False, sort = False).sum()
	df = df[(df['agree'] > df['disagree']) & (df['agree'] > 2)]
	df[['id', 'secret']] = df['url'].str.extract(r'http://farm[0-9]+\.staticflickr\.com/[0-9]+/(?P<id>[0-9]+)_(?P<secret>[a-z0-9]+)*')
	df['id'] = df['id'].astype('int64')
	df = df.set_index('id')
	df = df.sort_index()
	return df

def download(df, api_key, api_secret, fname = 'metadata.json'):
	print(f'Fetching flickr metadata...')
	
	flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
	df[['id', 'secret']] = df['url'].str.extract(r'http://farm[0-9]+\.staticflickr\.com/[0-9]+/(?P<id>[0-9]+)_(?P<secret>[a-z0-9]+)*')

	missing = []
	with open(os.path.join(p.data, fname), 'w') as f:
		nr_it, nr_img = 0, 0
		t = time.perf_counter()
		for r in df.itertuples():
			if (nr_it % 60 == 0):
				print(f'{nr_it} ({nr_img}) / {df.shape[0]}	QPI: { nr_it / (time.perf_counter() - t)}')
			try:
				photo = flickr.photos.getInfo(photo_id = r.id, secret = r.secret)
				response = requests.get(r.url)
				if 'jpeg' not in response.headers.get('content-type').lower():
					raise Exception(f'Resource at <{r.url}> is {response.headers.get("content-type").lower()} not JPEG.')
			except (flickrapi.exceptions.FlickrError, Exception) as e:
				missing.append(int(r.id))
				print(e)
			else:
				json.dump(photo, f)
				f.write('\n')
				with open(os.path.join(p.images, r.url.split('/')[-1]), 'wb') as img:
					img.write(response.content)
				
				nr_img += 1

			nr_it += 1
			time.sleep(0.8)

	print('Fetch completed.')
	return pd.Index(missing)


def build_subset(df, dir):
	
	with open(os.path.join(dir, 'images.map'), 'w') as f:
		for e in df['url'].tolist():
			f.write(os.path.join(p.images, e.split('/')[-1]))
			f.write('\n')

	for col in ['emotion', 'title', 'desc', 'tags']:
		with open(os.path.join(dir, f'{col}.txt'), 'wb') as f:
			for e in df[col].tolist():
				f.write(e.encode("utf-8"))
				f.write('\n'.encode("utf-8"))

def train_val_test_split(labels):
	X_train, X_tmp, y_train, y_tmp = train_test_split(labels.index, labels, test_size = 0.3, stratify = labels, random_state = 20)
	X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size = 0.5, stratify = y_tmp, random_state = 19)
	return (X_train, X_val, X_test)

def load_metadata(filename = 'metadata.json'):
	d = {'id' : [], 'json' : []}
	with open(os.path.join(p.data, filename)) as f:
		for line in f:
			photo = json.loads(line)
			d['id'].append(int(photo['photo']['id']))
			d['json'].append(photo)

	return pd.DataFrame(d).set_index('id')

def extract_features(df):
	feats = {}
	photo = df['json']
	feats['title'] = photo['photo']['title']['_content']
	feats['desc'] = photo['photo']['description']['_content']
	feats['tags'] = ' '.join([tag['_content'] for tag in photo['photo']['tags']['tag'] if tag['machine_tag'] == 0])
	
	patt = re.compile(r'<.*?>')
	crlf = re.compile(r'[\r\n]+')
	for k, v in feats.items():
		v = patt.sub('', v)
		v = html.unescape(v)
		v = crlf.sub(' ', v)
		feats[k] = v

	return feats


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('-k', '--key', required=True)
	parser.add_argument('-s', '--secret', required=True)
	parser.add_argument('-d', '--download', action = 'store_true')
	args = parser.parse_args()

	df = load_urls()
	if (args.download):
		download(df, args.key, args.secret)
	df2 = load_metadata().apply(extract_features, axis = 1, result_type = 'expand')
	df = df.join(df2, how = 'inner')
	
	masks = train_val_test_split(df['emotion'])

	df_mask = df.loc[df.index.join(masks[0], 'inner')]
	build_subset(df_mask, p.train)

	df_mask = df.loc[df.index.join(masks[1], 'inner')]
	build_subset(df_mask, p.val)

	df_mask = df.loc[df.index.join(masks[2], 'inner')]
	build_subset(df_mask, p.test)