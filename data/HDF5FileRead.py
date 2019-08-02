#!/usr/bin/python
import numpy as np
import pandas as pd
import os

def ADE20K_h5():
	list1 = os.listdir("./ADE20K/validation")
	#list2 = 
	for i in range(len(list1)):

		list1[i] = "data/ADE20K/validation/"+list1[i]
	#print(list1)

	data = {'path':list1}
        #data = {'semantic_map_path':list2}


	b = pd.DataFrame(data)
	print(b)
	h5 = pd.HDFStore('./ADE_paths_val.h5','w')
	h5['df'] = b
	h5.close()

	df = pd.read_hdf('./ADE_paths_val.h5', key='df').sample(frac=1).reset_index(drop=True).sort_values(by='path',ascending=False)
	print(df)

def ADE20K_h5_train():
	list1 = os.listdir("./ADE20K/training")
	#list2 = 
	for i in range(len(list1)):

		list1[i] = "data/ADE20K/training/"+list1[i]
	#print(list1)

	data = {'path':list1}
        #data = {'semantic_map_path':list2}


	b = pd.DataFrame(data)
	print(b)
	h5 = pd.HDFStore('./ADE_paths_train.h5','w')
	h5['df'] = b
	h5.close()

	df = pd.read_hdf('./ADE_paths_train.h5', key='df').sample(frac=1).reset_index(drop=True).sort_values(by='path',ascending=False)
	print(df)

def Kodak_h5(file_name):
	list1 = os.listdir("./Kodak/")
	for i in range(len(list1)):

		list1[i] = "data/Kodak/"+list1[i]
	#print(list1)

	data = {'path':list1}

	b = pd.DataFrame(data)
	print(b)
	h5 = pd.HDFStore(file_name,'w')
	h5['df'] = b
	h5.close()

	df = pd.read_hdf(file_name, key='df').sample(frac=1).reset_index(drop=True).sort_values(by='path',ascending=False)
	print(df)



def cityscapes_list():

	df = pd.read_hdf('./cityscapes_paths_train.h5', key='df').sample(frac=1).reset_index(drop=True).sort_values(by='path')
	print(df)

if __name__ == '__main__':
	#cityscapes_list()
	ADE20K_h5_train()
	#file_name = 'kodak.h5'
	#Kodak_h5(file_name)
	#print("File Generating Completed!: Kodak Dataset ===> {}".format(file_name))
