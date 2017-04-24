import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from data import dataProcess
from keras import backend as K

def dice_coef(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f*y_true_f) + K.sum(y_pred_f*y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
	return 1.-dice_coef(y_true, y_pred)


class multiNet(object):

	def __init__(self, img_rows = 256, img_cols = 256, label_num = 10):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.label_num = label_num

	def load_data(self):

		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data(reverse=True)
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test

	

	def get_model(self):
		
		'''
		using vgg-16
		'''

		inputs = Input((self.img_rows, self.img_cols,1))

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		#conv4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		#conv5 = Dropout(0.5)(conv5)
		pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)

		fc6 = Dense(4096, activation = 'relu')
		fc6 = Dropout(0.5)(fc6)

		fc7 = Dense(4096, activation = 'relu')
		fc7 = Dropout(0.5)(fc7)

		fc8 = Dense(self.label_num, activation = 'sigmoid')			

		model = Model(input = inputs, output = fc8)
		#model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics=[dice_coef,distance_loss])
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])																																												

		return model


	def train(self):

		print("loading data")
		imgs_train, train_label, imgs_test = self.load_data()
		print("loading data done")
		model = self.get_model()
		print("got multinet")

		model_checkpoint = ModelCheckpoint('multinet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, train_label, batch_size=10, nb_epoch=20, verbose=1, shuffle=True, callbacks=[model_checkpoint])

		print('predict test data')
		out = model.predict(imgs_test, batch_size=10, verbose=1)

		arr_threshold = np.zeros(out.shape[1]) + 0.5
		y_pred = np.array([[1 if out[i,j]>=arr_threshold[j] else 0 for j in range(imgs_test.shape[1])] for i in range(len(imgs_test))])
		np.save('out.npy', out)
		np.save('test_pred.npy', y_pred)


if __name__ == '__main__':
	mynet = multiNet()
	mynet.train()