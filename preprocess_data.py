import numpy as np
import os
from glob import glob
import shutil
import random

from download_images import loadData

def getDataLabels():
	data = loadData()
	_ = labelData(data)
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

def getImageName(directory="data/HOUSES"):
	images = os.listdir(directory)
	images = list(map(lambda x: x[:-4], images))
	return images


def splitTrainTestVal(data, image_names):
	image_names.sort()
	random.seed(231)
	random.shuffle(image_names)

	split_1 = int(0.6 * len(image_names))
	split_2 = int(0.8  * len(image_names))
	train = data[:split_1]
	val = data[split_1:split_2]
	test = data[split_2:]
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


def getImagesInFolderAll(data_list, data_names):
	for data, data_name in zip(data_list, data_names):
		getImagesInFolder(data=data, folder_name=data_name)

def filterData(data, image_names, row_id):
	data[row_id] = data[row_id].apply(lambda x: str(x))
	data = data[data[row_id].isin(image_names)]
	return data

def preprocessData(data):
	# Filter, split, and get images in folder
	image_names = getImageName()
	data = filterData(data, image_names, row_id="rowID")
	train, val, test = splitTrainTestVal(data, image_names)
	getImagesInFolderAll(data_list=[train, val, test], data_names=["train", "val", "test"])
	return train, val, test









