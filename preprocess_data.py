import numpy as np
import os
from glob import glob
import shutil
import random
import pandas as pd

from download_images import loadData

def loadData(file_name="data/clean_normalized.csv"):
	data = pd.read_csv(file_name)
	logTotalValue(data, column_id="TotalValue")
	return data


def loadDataResidual(file_name="data/residual_analysis.csv"):
	data = pd.read_csv(file_name)
	return data

def getClass(range_values):
    def inner(total_value):
        min_value = np.inf
        for i, r in enumerate(range_values):
            diff = np.abs(total_value - r)
            if diff < min_value:
                min_value = diff
                class_label = i
        return class_label
    return inner

def labelData(data):
	logTotalValue(data, column_id="TotalValue")
	mean_log_value = data["log_total_value"].mean()
	std_log_value = data["log_total_value"].std()
	num_classes_plus_2 = 8
	split_std = list(range(-int(num_classes_plus_2/2)+2,0)) + list(range(0, int(num_classes_plus_2/2)))
	range_values = list(map(lambda x: np.exp(x), list(map(lambda i: mean_log_value + i*std_log_value, split_std))))
	data["class_label"] = data["TotalValue"].apply(getClass(range_values))
	return range_values

def logTotalValue(data, column_id="TotalValue"):
	data["log_total_value"] = data[column_id].apply(lambda x: np.log(x))


def extractName(image_directory):
	list_names = image_directory.split("/")
	i = 0
	for i, e in enumerate(list_names):
		if e.endswith(".jpg"):
			idx = i
	return list_names[idx][:-4]


def renameImage(image_directory, old_image_name = 'gsv_0.jpg', destination_directory="data/HOUSES/"):
	if not os.path.exists(destination_directory):
		os.mkdir(destination_directory)

	image_new_name =  image_directory.split("/")[2] + ".jpg"
	inner_directory = os.listdir(image_directory)
	if old_image_name in inner_directory:
	    src_file = os.path.join(image_directory, old_image_name)
	    shutil.copy(src_file,destination_directory)

	    dst_file = os.path.join(destination_directory, old_image_name)
	    new_dst_file_name = os.path.join(destination_directory, image_new_name)
	    os.rename(dst_file, new_dst_file_name)

def renameImages(source_directory):
	image_directories = glob(source_directory + "/*/")
	list(map(renameImage, image_directories))


# Havent tested this function
def removeEmptyImages(image_directory, old_image_name = 'gsv_0.jpg'):
	inner_directory = os.listdir(image_directory)
	if old_image_name in inner_directory:
	  	pass
	else:
		shutil.rmtree(image_directory)

def getImageName(directories=["data/HOUSES"]):
	for i, dir_ in enumerate(directories):
		images = os.listdir(dir_)
		images = list(map(lambda x: x[:-4], images))
		images = set(images)
		if i == 0:
			total_images = set(images)
		else:
			total_images = total_images.intersection(images)

	return list(total_images)


def splitTrainTestVal(data, image_names, save=False):
	image_names.sort()
	random.seed(231)
	random.shuffle(image_names)

	split_1 = int(0.8 * len(image_names))
	split_2 = int(0.9  * len(image_names))
	train = data[:split_1]
	val = data[split_1:split_2]
	test = data[split_2:]
	if save:
		train.to_csv("data/CSVFiles/train.csv")
		val.to_csv("data/CSVFiles/val.csv")
		test.to_csv("data/CSVFiles/test.csv")
	return train, val, test


def getImagesInFolder(data, folder_name, source_location="data/HOUSES", images_folder="data/HOUSES_SPLIT"):
	if not os.path.exists(images_folder):
		os.mkdir(images_folder)

	cwd = os.getcwd()
	destination_directory = os.path.join(cwd, images_folder, folder_name)
	if os.path.exists(destination_directory):
		shutil.rmtree(destination_directory)

	os.mkdir(destination_directory)

	data["image_name"] = data["rowID"].apply(lambda x: str(x) + '.jpg')

	for image_name in data["image_name"]:
		src_file = os.path.join(source_location, image_name)
		shutil.copy(src_file,destination_directory)


def getImagesInFolderAll(data_list, data_names, source_location="data/HOUSES", images_folder="data/HOUSES_SPLIT"):
	for data, data_name in zip(data_list, data_names):
		getImagesInFolder(data=data, folder_name=data_name, source_location=source_location, images_folder=images_folder)

def filterData(data, image_names, row_id):
	data[row_id] = data[row_id].apply(lambda x: str(x))
	data = data[data[row_id].isin(image_names)]
	return data

def preprocessData(data, num_images=None, source_locations=["data/HOUSES"], images_folders=["data/HOUSES_SPLIT"]):
	# Filter, split, and get images in folder
	image_names = getImageName(directories=source_locations)
	data = filterData(data, image_names, row_id="rowID")
	if num_images:
		data = data.head(num_images)
		image_names = image_names[:num_images]
	train, val, test = splitTrainTestVal(data, image_names, save=False)

	for source_location, images_folder in zip(source_locations, images_folders):
		getImagesInFolderAll(data_list=[train, val, test], data_names=["train", "val", "test"],
			source_location=source_location, images_folder=images_folder)
	return train, val, test











