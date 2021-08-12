import os, sys
import pandas as pd
import torch

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def read_files(DATAPATH, dataset, verbose=True):
	"""Read the files based on whether the dataset exists in .txt or .h5 files
	DATAPATH: folder with the dataset files"""
	events = []
	for file in os.listdir(DATAPATH):
		if dataset in file:
			if verbose:
				print("Reading data from {}".format(file))
				events = pd.read_hdf(os.path.join(DATAPATH, file), key='table').values

	return events

def Loader(dataset, batch_size, test):

	datapath = './data/top_tagging'
	data = read_files(datapath, dataset)	

	if test == True:
		split = int(len(data) * 0.01)
	else:	
		split = int(len(data) * 0.01)

	events=data
	events_train = events[:split]
	shape = events_train.shape[1]

	print(events_train.shape)

	"""Prepare train and validate data loaders"""
	train_loader = torch.utils.data.DataLoader(
			torch.from_numpy(events_train).to(device),
			batch_size = batch_size,
			shuffle = True,
			drop_last = True,
			)
	
	return train_loader, split, shape
