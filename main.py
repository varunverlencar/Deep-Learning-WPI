#Varun Verlencar
#vvverlencar@wpi.eduu
#WPI-Hand Gestures
import numpy
from keras.callbacks import ModelCheckpoint
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
import os
import h5py
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

def ensure_dir(f):
	d = os.path.dirname(f)
	if not os.path.exists(d):
 		os.makedirs(d)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing
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

print "training data read"

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
	'dataset/validation',
	target_size=(224, 224),
	batch_size=10,
	shuffle = True,
	class_mode='categorical')

print "validation data read"

# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(
	'dataset/test',
	target_size=(224, 224),
	batch_size=10,
	shuffle = True,
	class_mode='categorical')

print "validation data read"

learn_r= 0.00001
dec = 0.0000005
reg = 0.0001

# define a simple CNN model
def baseline_model():
	# create model
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))     #0
	first_layer = model.layers[-1]
	# this is a placeholder tensor that will contain our generated images
	input_img = first_layer.input

	model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1',W_regularizer = l2(reg)))
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

i=5000 #samples_per_epoch
j=1600 #nb_val_samples

folder  = "Weights/Best/main/"
ensure_dir(folder)

filepath=folder +"weights.2best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


print 'fitting model'
history = model.fit_generator(
	train_generator,
	samples_per_epoch=i,
	nb_epoch=50,
	validation_data=validation_generator,
	nb_val_samples=j,
	verbose = 2,
	callbacks = callbacks_list
	)

folder  = "Weights/main/"
ensure_dir(folder)
model.save_weights( folder +'2_try.h5')

vscores = model.evaluate_generator(validation_generator,val_samples = j)
print("Validation Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-vscores[1]*100,j,i))

tscores = model.evaluate_generator(test_generator,
	val_samples = j)
print("Test Error: %.2f%%, for nb_val_samples=%d samples_per_epoch=%d" % (100-tscores[1]*100,j,i))

folder  = "Images/main/"
ensure_dir(folder)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
fileName = "2_accuracy.png"
plt.savefig(folder + fileName, bbox_inches='tight')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
fileName = "2_loss.png" 
plt.savefig(folder + fileName, bbox_inches='tight')
