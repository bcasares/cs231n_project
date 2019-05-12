import google_streetview.api
import pandas as pd
from ast import literal_eval
import os
import urllib.request, urllib.parse

# key = "&key=" + "AIzaSyA7kqDO_6lbirEKKqDHcCIl05-AR0t9Yqg" # homework account
# key = "&key=" + "AIzaSyDl3mMCFcSapH5j0TeAM97Ua_noRLiZKJY" # project account

def loadData(file_name="data/CSVFiles/LAProp_Residential_2017_Only_Houses.csv"):
	data = pd.read_csv(file_name)
	return data


def changeParams(names, values):
	params = [{
		'size': '600x600', # max 640x640 pixels
		'location': "",
		'heading': '0',
		'pitch': '0',
		'key': 'AIzaSyDl3mMCFcSapH5j0TeAM97Ua_noRLiZKJY'
	}]
	for name, value in zip(names, values):
		params[0][name] = value
	return params


def getImageGoogleMap(data,location_name,row_name):
	data[location_name] = data[location_name].astype(str)
	def inner(lat_long):
		save_name = str(data[data[location_name] == lat_long].iloc[0][row_name])
		download_directory = "data/DOWNLOAD_HOUSES"
		dir_path = download_directory + '/' + save_name
		if os.path.isdir(dir_path):
			return
		lat_long = literal_eval(lat_long)
		lat_long = str(lat_long[0]) + "," + str(lat_long[1])
		params = changeParams(names=["location"], values=[lat_long])
		results = google_streetview.api.results(params)
		results.download_links(dir_path=dir_path)
		results.save_links(dir_path + '/links.txt')
	return inner


