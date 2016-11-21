from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

import cv2, numpy as np
import pickle
    
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

    
    return model

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    the_filename='HandGesture.txt'
    #with open(the_filename, 'wb') as f:
    #    pickle.dump(my_list, f)
    # with open(the_filename, 'rb') as f:
    #     my_list = pickle.load(f)
    # im = cv2.resize(cv2.imread('000.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)

    learn_r= 0.0001
    dec = 0.0000005
    reg = 0.000001
    # Test pretrained model
    model = baseline_model('weights.best.hdf5')
    opt = Adam(lr=learn_r, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=dec)
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    out = model.predict(im)
    #print np.argmax(out)
    #print my_list[np.argmax(out)]
    
    
    
    ########################
    # REAL-TIME PREDICTION #
    ########################

    print '... Initializing RGB stream'
    
     #### Initialize built-in webcam
    cap = cv2.VideoCapture(0)
    # Enforce size of frames
    cap.set(3, 320) 
    cap.set(4, 240)

    shot_id = 0
 
    #### Start video stream and online prediction
    while (True):
         # Capture frame-by-frame
    
#        start_time = time.clock()
        
        ret, frame = cap.read()
        
        #color_frame = color_stream.read_frame() ## VideoFrame object
        #color_frame_data = frame.get_buffer_as_uint8() ## Image buffer
        #frame = convert_frame(color_frame_data, np.uint8) ## Generate BGR frame
                
        im = cv2.resize(frame, (224, 224)).astype(np.float32)
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)
        
        out = model.predict(im)
        #print np.argmax(out)
        #print my_list[np.argmax(out)]
        
        # we need to keep in mind aspect ratio so the image does
        # not look skewed or distorted -- therefore, we calculate
        # the ratio of the new image to the old image
        #r = 100.0 / frame.shape[1]
        dim = (640, 480)
 
        # perform the actual resizing of the image and show it
        resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(resized,my_list[np.argmax(out)],(20,450), font, 1, (255,255,255),1,1)
        # Display the resulting frame
        cv2.imshow('DeepNN-ABB',resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()