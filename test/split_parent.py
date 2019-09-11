import os
import shutil
import random
from math import floor
import argparse



def get_file_list_from_dir(datadir):
    all_files = os.listdir(os.path.abspath(datadir))
    #data_files = list(filter(lambda file: file.endswith('.data'), all_files))
    #get_training_and_testing_sets(all_files)
    #create_train_val_test_dir(datadir)
    return all_files


def create_train_val_test_dir(datadir,curr_dir):
	if not os.path.exists(datadir+'/'+"train"):
		os.makedirs(datadir+'/'+"train")
		os.makedirs(datadir+'/'+"test")
		os.makedirs(datadir+'/'+"val")
	if not os.path.exists(datadir+'/train'+'/'+ curr_dir):
		os.makedirs(datadir+'/train'+'/'+ curr_dir)
		#os.makedirs(datadir+'/test'+'/'+ curr_dir)
		os.makedirs(datadir+'/val'+'/'+ curr_dir)


def randomize_files(file_list):
	random.shuffle(file_list)

def get_training_and_testing_sets(file_list):
	testSplit = 0
	valSplit = 0.2
	trainSplit = 1-(testSplit+valSplit)
	train_split_index = int(floor(len(file_list) * trainSplit))
	print(train_split_index)
	test_split_index = train_split_index + int(floor( len(file_list) * testSplit ))
	trainData = file_list[:train_split_index]
	testData = file_list[train_split_index:test_split_index]
	valData = file_list[test_split_index:]
	return trainData,testData,valData

def move_files_to_train_test_folders(train,test,val,datadir,curr_dir):
	for x in train:
		os.rename(datadir+'/'+curr_dir+'/'+x,datadir+'/'+'train/'+curr_dir+'/'+x)
	for x in test:
		os.rename(datadir+'/'+curr_dir+'/'+x,datadir+'/'+'test/'+curr_dir+'/'+x)
	for x in val:
		os.rename(datadir+'/'+curr_dir+'/'+x,datadir+'/'+'val/'+curr_dir+'/'+x)
	os.rmdir(datadir+'/'+curr_dir)

def get_folders_list(dataPath):
	#list of all folders whose names are classes from given directory
	dirs = [d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))]
	return dirs
	
#If only particular number of files to be put in test and val folders
def get_x_file_in_val_test(file_list):
	testSplit = 0
	valSplit = 1
	#trainSplit = 1-(testSplit+valSplit)
	train_split_index = len(file_list) - valSplit - testSplit
	#print(train_split_index)
	test_split_index = train_split_index + testSplit
	trainData = file_list[:train_split_index]
	testData = file_list[train_split_index:test_split_index]
	valData = file_list[test_split_index:]
	return trainData,testData,valData

def set_sample_files(curr_dir,dataDir,file_list,num_train,num_val):
	random.seed()
	print(curr_dir)
	print(len(file_list))
	population=list(range(len(file_list)))
	sample=random.sample(population,num_train+num_val)
	for i in range(num_train):
		shutil.copyfile(dataDir+'/train/'+curr_dir+'/'+file_list[sample[i]],dataDir+'/sample/train/'+curr_dir+'/'+file_list[sample[i]])
	for i in range(num_val):
		shutil.copyfile(dataDir+'/train/'+curr_dir+'/'+file_list[sample[i+num_train]],dataDir+'/sample/val/'+curr_dir+'/'+file_list[sample[i+num_train]])


def create_sample_dir(dataDir,num_train,num_val):
	dir_list = get_file_list_from_dir(dataDir+'/train')
	for dir in dir_list:
		file_list=os.listdir(dataDir+'/train/'+dir)
		#create_train_val_test_dir(dataDir+'/sample',dir)
		randomize_files(file_list)
		set_sample_files(dir,dataDir,file_list,num_train,num_val)
		
#print(train)
#print(test)
#print(val)
'''
dirList=get_folders_list('/tmp/Data_DeltaKL/CV_materials/Bulbs_data')
print(dirList)

for di in dirList:
	file_list = get_file_list_from_dir(di)
	randomize_files(file_list)
	train,test,val=get_training_and_testing_sets(file_list)
	move_files_to_train_test_folders(train,test,val,di)
'''

if __name__=="__main__":
	a=argparse.ArgumentParser()
	a.add_argument("--dataFolder")
	args = a.parse_args()
	if args.dataFolder is None:
		a.print_help()
		sys.exit(1)

	if (not os.path.exists(args.dataFolder)):
		print("directory does not exsist")
		sys.exit(1)

	dir_list = get_file_list_from_dir(args.dataFolder)
	for dir in dir_list:
		file_list=os.listdir(args.dataFolder+'/'+dir)
		create_train_val_test_dir(args.dataFolder,dir)
		randomize_files(file_list)
		train,test,val=get_training_and_testing_sets(file_list)
		#train,test,val=get_x_file_in_val_test(file_list)
		move_files_to_train_test_folders(train,test,val,args.dataFolder,dir)
	
