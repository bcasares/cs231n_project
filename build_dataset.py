"""Split the HOUSES dataset into train/val/test and resize images to 64x64.
"""

import argparse
import random
import os

from PIL import Image, ImageFile
from tqdm import tqdm



def resize_and_save(filename, output_dir, size):
	"""Resize the image contained in `filename` and save it to the `output_dir`"""

	ImageFile.LOAD_TRUNCATED_IMAGES = True

	image = Image.open(filename)
	# Use bilinear interpolation instead of the default "nearest neighbor" method
	image = image.resize((size, size), Image.BILINEAR)
	image.save(os.path.join(output_dir, filename.split('/')[-1]))


def resizeAndSave(data_dir, output_dir, SIZE = 64):

	assert os.path.isdir(data_dir), "Couldn't find the dataset at {}".format(data_dir)

	# Define the data directories
	train_data_dir = os.path.join(data_dir, 'train')
	test_data_dir = os.path.join(data_dir, 'test')
	val_data_dir = os.path.join(data_dir, 'val')


	# Get the filenames in each directory (train and test)
	train_filenames = os.listdir(train_data_dir)
	train_filenames = [os.path.join(train_data_dir, f) for f in train_filenames if f.endswith('.jpg')]

	test_filenames = os.listdir(test_data_dir)
	test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

	val_filenames = os.listdir(val_data_dir)
	val_filenames = [os.path.join(val_data_dir, f) for f in val_filenames if f.endswith('.jpg')]


	filenames = {'train': train_filenames,
	             'val': val_filenames,
	             'test': test_filenames}

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)

    # Preprocess train, val and test
	for split in ['train', 'val', 'test']:
		output_dir_split = os.path.join(output_dir, '{}'.format(split))
		if not os.path.exists(output_dir_split):
			os.mkdir(output_dir_split)
		else:
			print("Warning: dir {} already exists".format(output_dir_split))

		print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
		for filename in tqdm(filenames[split]):
			resize_and_save(filename, output_dir_split, size=SIZE)

		print("Done building dataset")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', default="data/HOUSES_SPLIT", help="Directory with the SIGNS dataset")
	parser.add_argument('--output_dir', default="data/HOUSES_SPLIT_64_64", help="Where to write the new data")

	args = parser.parse_args()
	resizeAndSave(data_dir=args.data_dir, output_dir =args.output_dir, SIZE = 64)


