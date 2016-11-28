from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.regularizers import l2, activity_l2
from keras.optimizers import Nadam
import cv2, numpy as np
import pickle

def output_label(out): 
    if out == 0:
        return 'Gesture-1'
    elif out == 1:
        return 'Gesture-2'
    elif out == 2:
        return 'Gesture-3'
    elif out == 3:
        return 'Gesture-4'
    elif out == 4:
        return 'Gesture-5'

def baseline_model(weights_path=None):
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
    model.add(Dense(128 , activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128 , activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(5, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    the_filename='HandGesture.txt'
    #with open(the_filename, 'wb') as f:
    #    pickle.dump(my_list, f)
    # with open(the_filename, 'rb') as f:
    #     my_list = pickle.load(f)
    im = cv2.resize(cv2.imread('testImages/Gesture5.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    learn_r= 0.0001
    dec = 0.000000
    reg = 0.000000001
    # Test pretrained model
    model = baseline_model('Aug/Weights/Best/main/Aug_weights4.best.hdf5')
    # opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
    opt = Nadam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=dec)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    out = model.predict(im)
    print out
    print np.argmax(out)
    print output_label(np.argmax(out))
