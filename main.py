# Carlos Morato, PhD.
# cwmorato@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Simple CNN for the MNIST Dataset
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
from keras.layers.noise import GaussianNoise
from keras.preprocessing.image import ImageDataGenerator
import csv
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'dataset/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
	'dataset/train',  # this is the target directory
	target_size=(224, 224),
	batch_size=10,
	shuffle = True,
	class_mode='categorical')  

print "read training"
# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
	'dataset/validation',
	target_size=(224, 224),
	batch_size=10,
	shuffle = True,
	class_mode='categorical')

print "read validation"

# load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # reshape to be [samples][channels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# # normalize inputs from 0-255 to 0-1
# X_train = X_train / 255
# X_test = X_test / 255
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

learn_r= 0.0000001
dec = 0.000000005
reg = 0
# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))     #0
	first_layer = model.layers[-1]
	# this is a placeholder tensor that will contain our generated images
	input_img = first_layer.input

	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.3))

	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
	model.add(ZeroPadding2D((1, 1)))
	model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Dropout(0.3))

	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
	# model.add(ZeroPadding2D((1, 1)))
	# model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
	# model.add(MaxPooling2D((2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128	, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128	, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(5, activation='softmax'))

	opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
print "model built"
print model.summary()

i=10
j=15
# Fit the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=j, batch_size=i,shuffle=True, verbose=2)

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print 'fitting model'
history = model.fit_generator(
	train_generator,
	samples_per_epoch=2000,
	nb_epoch=50,
	validation_data=validation_generator,
	nb_val_samples=800,
	verbose = 2
	callback = callbacks_list)

model.save_weights('first_try.h5')

scores = evaluate_generator(validation_generator,
	val_samples = 2000)
print("Error: %.2f%%, for nb_epoch=%d batch_size=%d" % (100-scores[1]*100,j,i))

# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("CNN Error: %.2f%%, for nb_epoch=%d batch_size=%d" % (100-scores[1]*100,j,i))

# csvfile =  open('.csv', 'a')
# fieldnames = [' Error', 'Epoch','Batch']
# writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
# writer.writeheader()
# writer.writerow({'Baseline Error': 100-scores[1]*100,'Epoch': j,'Batch':i})

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig("First_accuracy.png", bbox_inches='tight')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig("First_loss.png", bbox_inches='tight')
plt.show()