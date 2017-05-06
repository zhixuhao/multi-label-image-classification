from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import pandas as pd
import os
import glob
import skimage.io as io
from tqdm import tqdm

df = pd.read_csv('./data/train_tap.csv')
npdf = df.values[:,:]
np.random.shuffle(npdf)
train_num = 35000
train_df = npdf[0:train_num]
val_df = npdf[train_num:]

tdf = pd.read_csv('./data/sample_submission.csv')
nptdf = tdf.values[:,:]

def strToarr(strin):

	'''
	convert str to nparray
	ex. '[0 1 0 1 0]' => [0,1,0,1,0]
	'''
	strarr = strin[1:len(strin)-1].split(" ")
	for i in range(len(strarr)):
		strarr[i] = int(strarr[i])
	return np.array(strarr)

train_data = np.ndarray((train_num,256,256,4),dtype = np.float32)
train_label = np.ndarray((train_num,17),dtype = np.uint8)
train_err = []
train_err_arr = []
i = 0
ft = open('./data/train_err.txt','w')
for row in tqdm(train_df):
    try:
        imgname = row[0]
        strlabel = row[1]
        imgname = imgname.replace(' ','_')
        img3 = load_img('./data/train-jpg/'+imgname+'.jpg')
        img3 = img_to_array(img3)
        img4 = io.imread('./data/train-tif/'+imgname+'.tif')
        img4 = img4[:,:,3].astype('float32')
        train_data[i,:,:,0:3] = img3/255
        train_data[i,:,:,3] = img4/np.power(2,16)
        train_label[i] = strToarr(strlabel)
    except:
    	train_err_arr.append(i)
        train_err.append(row[0])
        ft.write(row[0] + '\n')
        #train_data = np.delete(train_data,i,0)
        #train_label = np.delete(train_label,i,0)
    i += 1
ft.close()
train_data = np.delete(train_data,train_err_arr,0)
train_label = np.delete(train_label,train_err_arr,0)
np.save('./data/npydata/train_data.npy', train_data)
np.save('./data/npydata/train_label.npy', train_label)
print train_data.shape
print train_label.shape
print train_err

val_data = np.ndarray((40479-train_num,256,256,4),dtype = np.float32)
val_label = np.ndarray((40479-train_num,17),dtype = np.uint8)
val_err = []
val_err_arr = []
i = 0
fv = open('./data/val_err.txt','w')
for row in tqdm(val_df):
    try:
        imgname = row[0]
        strlabel = row[1]
        imgname = imgname.replace(' ','_')
        img3 = load_img('./data/train-jpg/'+imgname+'.jpg')
        img3 = img_to_array(img3)
        img4 = io.imread('./data/train-tif/'+imgname+'.tif')
        img4 = img4[:,:,3].astype('float32')
        val_data[i,:,:,0:3] = img3/255
        val_data[i,:,:,3] = img4/np.power(2,16)
        val_label[i] = strToarr(strlabel)
    except:
        val_err.append(row[0])
        val_err_arr.append(i)
        fv.write(row[0]+'\n')
        #val_data = np.delete(val_data,i,0)
        #val_label = np.delete(val_label,i,0)
    i += 1
fv.close()
val_data = np.delete(val_data,val_err_arr,0)
val_label = np.delete(val_label,val_err_arr,0)
np.save('./data/npydata/val_data.npy', val_data)
np.save('./data/npydata/val_label.npy', val_label)
print val_data.shape
print val_label.shape
print val_err

test_data = np.ndarray((40669,256,256,4),dtype = np.float32)
test_err = []
#test_err_val = []
i = 0
fe = open('./data/test_err.txt','w')
for row in tqdm(nptdf):
    try:
        imgname = row[0]
        imgname = imgname.replace(' ','_')
        img3 = load_img('./data/train-jpg/'+imgname+'.jpg')
        img3 = img_to_array(img3)
        img4 = io.imread('./data/train-tif/'+imgname+'.tif')
        img4 = img4[:,:,3].astype('float32') 
        test_data[i,:,:,0:3] = img3/255
        test_data[i,:,:,3] = img4/np.power(2,16)
    except:
        test_err.append(row[0])
        fe.write(row[0]+'\n')
        #test_err_val.append(i)
        #test_data = np.delete(test_data,i,0)
    i += 1    
fe.close()
#test_data = np.delete(val_data,val_err_arr,0)
np.save('./data/npydata/test_data.npy', test_data)
print test_data.shape
print test_err