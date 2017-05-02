from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import pandas as pd
import os
import glob
from libtiff import TIFF

class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "./data/train-jpg", label_path = "./data", test_path = "./data/test-jpg", npy_path = "./data/npydata", img_type = "jpg", channels = 4, num_class = 17):

		"""
		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path
		self.num_class = num_class
		self.channels = channels

	def strToarr(self, strin):

		'''
		convert str to nparray
		ex. '[0 1 0 1 0]' => [0,1,0,1,0]
		'''
		strarr = strin[1:len(strin)-1].split(" ")
		for i in range(len(strarr)):
			strarr[i] = int(strarr[i])
		return np.array(strarr)


	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.num_class), dtype=np.uint8)
		df = pd.read_csv(self.label_path + '/train_tap.csv')
		npdf = df.values[:,1]
		for i in range(len(imgs)):
			midname = 'train_' + str(i) + '.' + self.img_type
			img = load_img(self.data_path + "/" + midname)
			img = img_to_array(img)
			imgdatas[i] = img
			imglabels[i] = self.strToarr(npdf[i])
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_label_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for i in range(len(imgs)):
			midname = 'test_' + str(i) + '.' + self.img_type
			img = load_img(self.test_path + "/" + midname)
			img = img_to_array(img)
			imgdatas[i] = img
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_label_train = np.load(self.npy_path+"/imgs_label_train.npy")
		imgs_train = imgs_train.astype('float32')
		#imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		mean = imgs_train.mean(axis = 0)
		np.save(self.npy_path + '/imgs_train_mean.npy', mean)
		imgs_train -= mean	
		return imgs_train,imgs_label_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		mean = imgs_test.mean(axis = 0)
		np.save(self.npy_path + '/imgs_test_mean.npy', mean)
		imgs_test -= mean	
		return imgs_test

	def generator(self, batch_size = 16, shuffle = True):
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		df = pd.read_csv(self.label_path + '/train_tap.csv')
		npdf = df.values[:,1]
		num_imgs = len(imgs)
		index_array = np.arange(num_imgs)
		if shuffle:
			index_array = np.random.permutation(num_imgs)
		curindex = 0
		while True:
			arr = index_array[curindex:curindex+batch_size]
			batch_x = np.zeros((batch_size,self.out_rows,self.out_cols,self.channels))
			batch_y = np.zeros((batch_size,self.num_class))
			for index,value in enumerate(arr):
				midname = 'train_' + str(value) + '.' + self.img_type
				img = TIFF.open(self.data_path+"/"+midname)
				img = img.read_image()
				img = np.array(img)
				batch_x[index] = img
				batch_y[index] = self.strToarr(npdf[value])
			if num_imgs > curindex + batch_size:
				curindex += batch_size
			else:
				curindex = num_imgs - batch_size
			yield (batch_x,batch_y)





if __name__ == "__main__":

	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(256,256)
	mydata.create_train_data()
	mydata.create_test_data()
	imgs_train,imgs_label_train = mydata.load_train_data()
	print imgs_train.shape,imgs_label_train.shape
	print imgs_label_train[0]
	print imgs_train[0]
