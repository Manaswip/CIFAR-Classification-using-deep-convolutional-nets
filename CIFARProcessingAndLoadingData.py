import cv2
import numpy as np
import cPickle	

""" CIFAR-10 dataset contains 60000 images out of which 10000 images are testing data in 6 different batches.
and each image contains RGB values

In this file data from all the batches is retrieved and converted to gray

Returns training data, validation data, and testing data
"""

def load_data_shared():
	training_array = np.zeros((50000,1024),dtype=np.uint8)
	training_labels = np.zeros((50000,),dtype=np.uint8)
	testing_array = np.zeros((10000,1024),dtype=np.uint8)
	testing_labels = np.zeros((10000,),dtype=np.uint8)
#Extracting training data and validaton data
	for d in xrange(5):
		folder = open('/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/CIFAR/data_batch_'+str(d+1),'rb')
		dictionary = cPickle.load(folder)
		data = dictionary['data']
		training_labels[d*10000:(d+1)*10000]=dictionary['labels']
		red = data[:,0:1024]
		blue = data[:,1024:2048]
		green = data[:,2048:]
		images=[]
		final_images=[]
#In opencv colored image is in BGR as opposed to RGB
#Get BGR values of image and then convert into Gray scale image
		for i in xrange(data.shape[0]):
			for j in xrange(32):
				image = [(b,g,r) for b,g,r in zip(blue[i,32*j:32*(j+1)],green[i,32*j:32*(j+1)],red[i,32*j:32*(j+1)])]
				images.append(image)
			training_array[d*10000+i] = np.ravel(cv2.cvtColor(np.asarray(images),cv2.COLOR_BGR2GRAY))
			images=[]

#Dividing 50000 images into 40000 training images 10000 validation images
	training_data = (training_array[0:40000],training_labels[0:40000])
	validation_data = (training_array[40000:50000],training_labels[40000:50000])
#Extracting testing data	
	folder = open('/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/CIFAR/test_batch','rb')
	dictionary = cPickle.load(folder)
	data = dictionary['data']
	testing_labels=dictionary['labels']
	red = data[:,0:1024]
	blue = data[:,1024:2048]
	green = data[:,2048:]
	images=[]
	final_images=[]
	for i in xrange(data.shape[0]):
		for j in xrange(32):
			image = [(b,g,r) for b,g,r in zip(blue[i,32*j:32*(j+1)],green[i,32*j:32*(j+1)],red[i,32*j:32*(j+1)])]
			images.append(image)
		testing_array[i] = np.ravel(cv2.cvtColor(np.asarray(images),cv2.COLOR_BGR2GRAY))
		images=[]

	testing_data = (testing_array,testing_labels)
	
	return training_data,validation_data,testing_data