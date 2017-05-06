import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
from data import dataProcess
from keras import backend as K
from sklearn.metrics import fbeta_score


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())



class multiNet(object):

	def __init__(self, img_rows = 256, img_cols = 256, label_num = 17):

		self.img_rows = img_rows
		self.img_cols = img_cols
		self.label_num = label_num

	def load_train_data(self):

		#mydata = dataProcess(self.img_rows, self.img_cols)
		#imgs_train, imgs_label_train = mydata.load_train_data()
		#imgs_test = mydata.load_test_data()
		imgs_train = np.load('./data/npydata/train_data.npy')
		imgs_label_train = np.load('./data/npydata/train_label.npy')
		mean = imgs_train.mean(axis = 0)
		np.save('./data/npydata/imgs_train_mean.npy', mean)
		imgs_train -= mean	
		return imgs_train, imgs_label_train#, imgs_test

	def load_val_data(self):

		#mydata = dataProcess(self.img_rows, self.img_cols)
		#imgs_train, imgs_label_train = mydata.load_train_data()
		#imgs_test = mydata.load_test_data()
		imgs_val = np.load('./data/npydata/train_val.npy')
		imgs_label_val = np.load('./data/npydata/val_label.npy')
		mean = imgs_val.mean(axis = 0)
		np.save('./data/npydata/imgs_val_mean.npy', mean)
		imgs_val -= mean	
		return imgs_val, imgs_label_val#, imgs_test

	def load_test_data(self):

		#mydata = dataProcess(self.img_rows, self.img_cols)
		#imgs_test = mydata.load_test_data()
		imgs_test = np.load('./data/npydata/test_data.npy')
		mean = imgs_test.mean(axis = 0)
		np.save('./data/npydata/imgs_test_mean.npy', mean)
		imgs_test -= mean	
		return imgs_test

	

	def get_model(self):
		
		'''
		using vgg-16
		'''

		inputs = Input((self.img_rows, self.img_cols,4))

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

		pool5 = Flatten()(pool5)

		fc6 = Dense(4096, activation = 'relu')(pool5)
		fc6 = Dropout(0.5)(fc6)

		fc7 = Dense(4096, activation = 'relu')(fc6)
		fc7 = Dropout(0.5)(fc7)

		fc8 = Dense(self.label_num, activation = 'sigmoid')(fc7)			

		model = Model(input = inputs, output = fc8)
		#model.compile(optimizer = Adam(lr = 1e-4), loss = dice_coef_loss, metrics=[dice_coef,distance_loss])
		model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics=[fbeta])																																												

		return model


	def train(self):

		print("loading data")
		imgs_train, train_label = self.load_val_data()
		print("loading train data done")
		imgs_val, val_label = self.load_val_data()
		print("loading val data done")
		model = self.get_model()
		print("got multinet")

		model_checkpoint = ModelCheckpoint('multinet.hdf5', monitor='loss',verbose=1, save_best_only=True)
		print('Fitting model...')
		model.fit(imgs_train, train_label, batch_size=32, nb_epoch=20, verbose=1, shuffle=True, validation_data=(imgs_val, val_label), callbacks=[model_checkpoint])

	def test(self):

		print("loading data")
		imgs_test = self.load_test_data()
		print("loading data done")

		model = self.get_model()
		model.load_weights('multinet.hdf5')
		print('predict test data')
		out = model.predict(imgs_test, batch_size=10, verbose=1)

		arr_threshold = np.zeros(out.shape[1]) + 0.5
		#arr_threshold = self.find_best_threshold(imgs_train, train_label)

		#y_pred = np.array([[1 if out[i,j]>=arr_threshold[j] else 0 for j in range(imgs_test.shape[1])] for i in range(len(imgs_test))])
		np.save('out.npy', out)
		#np.save('test_pred.npy', y_pred)


	def find_best_threshold(self, imgs_train, train_label):

		out = model.predict(imgs_train, batch_size=10, verbose=1)
		threshold = np.arange(0.1,0.9,0.1)
		acc = []
		accuracies = []
		best_threshold = np.zeros(train_label.shape[1])
		for i in range(out.shape[1]):
			y_prob = np.array(out[:,i])
			for j in threshold:
				y_pred = np.array([1 if prob>=j else 0 for prob in y_prob])
				acc.append( matthews_corrcoef(y_test[:,i],y_pred))
			acc   = np.array(acc)
			index = np.where(acc==acc.max()) 
			accuracies.append(acc.max()) 
			best_threshold[i] = threshold[index[0][0]]
			acc = []
		print "best_threshold", best_threshold
		print "accuracies", accuracies
		np.save('best_threshold.npy', best_threshold)
		return best_threshold



if __name__ == '__main__':
	mynet = multiNet()
	#model = mynet.get_model()
	mynet.train()
	#mynet.test()