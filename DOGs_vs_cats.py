import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR='/Users/tapan/Documents/SD_Dogs_vs_Cats_CNN/train'
TEST_DIR='/Users/tapan/Documents/SD_Dogs_vs_Cats_CNN/test1'
IMG_SIZE=50
LR=1e-3                       #learning rate=0.001

MODEL_NAME='dogsvscats-{}-{}.model'.format(LR,'6conv-basic-video')

def label_img(img):
     word_label=img.split('.')[0]
     if word_label == 'cat':
          return [1,0]
     elif word_label == 'dog':
          return [0,1] 


def create_train_data():
     training_data=[]
     for img in tqdm(os.listdir(TRAIN_DIR)):
          label=label_img(img)
          path=os.path.join(TRAIN_DIR,img)
          img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
          training_data.append([np.array(img),np.array(label)])
     shuffle(training_data)
     np.save('train_data.npy',training_data)
     return training_data 


def process_test_data():
     testing_data=[]
     for img in tqdm(os.listdir(TEST_DIR)):
          path=os.path.join(TEST_DIR,img)
          img_num=img.split('.')[0]
          img=cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(IMG_SIZE,IMG_SIZE))
          testing_data.append([np.array(img) , img_num])
     np.save('test_data.npy',testing_data)
     return testing_data


#train_data=create_train_data()
#if train_data is created before:
train_data=np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
# http://tflearn.org/layers/conv/
# http://tflearn.org/activations/
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)  

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR , loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet,tensorboard_dir='/Users/tapan/Documents/SD_Dogs_vs_Cats_CNN/log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('Model Loaded!')

train=train_data[:-500]
test=train_data[-500:]

X=np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y=[i[1] for i in train]

test_X=np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y=[i[1] for i in test]

model.fit({'input': X}, {'targets': y}, n_epoch=5, validation_set=({'input': test_X}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

#tensorboard --logdir=/Users/tapan/Documents/SD_Dogs_vs_Cats_CNN/log
model.save(MODEL_NAME)

import matplotlib.pyplot as plt

#if you dont have test data:
test_data=process_test_data()
#if you already have it:
#test_data=np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[-12:]):
    #cat=[1,0]
    #dog=[0,1]
    img_data=data[0]
    img_num=data[1]
    
    y=fig.add_subplot(3,4,num+1)
    
    orig=img_data
    data=img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    
    #we are interested in cats only
    model_out=model.predict([data])[0]
    
    if np.argmax(model_out)==1:
        str_label='Dog'
    else:
        str_label='Cat'
        
    
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.show()

