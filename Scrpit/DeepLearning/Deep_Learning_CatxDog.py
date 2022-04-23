# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: demetrios
"""


#%% [1] Import necessarios
import numpy as np 
import pandas as pd 
from os import listdir
from os.path import join, basename
from PIL import Image
#%% [2] Configuracoes da rede

IMG_HEIGHT = 50
IMG_WIDTH = 50
NUM_CHANNELS = 3

from threading import current_thread, Thread, Lock
from multiprocessing import Queue
# Any results you write to the current directory are saved as output.

#%% [3]  Inicializando as configuracoes da rede

batch_size = 500
num_train_images = 25000
num_test_images = 12500
num_train_threads = int(num_train_images/batch_size)  # 50
num_test_threads = int(num_test_images/batch_size)    # 25
lock = Lock()
#%% [4]  Usando fila para coletar os dados

def initialize_queue():
    queue = Queue()
    return queue

#%% [5] Setando a base dados para teste e treino
train_dir_path = 'dogs-vs-cats/train/'
test_dir_path = 'dogs-vs-cats/test1/'

train_imgs = [join(train_dir_path,f) for f in listdir(train_dir_path)]
test_imgs = [join(test_dir_path,f) for f in listdir(test_dir_path)]
print(len(train_imgs))
print(len(test_imgs))
#%% [6] Pegando o nome das imagens para configurar as categorias
def get_img_label(fpath):
    category = fpath.split(".")[-3]
    if category == "dog":
        return [1,0]
    elif category == "cat":
        return [0,1]

#%%
def get_img_array_labels(fpaths, queue):
    img_array = None
    labels = []
    for f in fpaths:
        arr = Image.open(f)
        arr = arr.resize((IMG_HEIGHT,IMG_WIDTH), Image.ANTIALIAS)
        arr = np.reshape(arr, (-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        if img_array is None:
            img_array = arr
        else:
            img_array = np.vstack((img_array, arr))
        labels.append(get_img_label(basename(f)))
    labels = np.array(labels)
    queue.put((img_array, labels))
#%%
def get_img_array(fpaths, queue):
    img_array = None
    for f in fpaths:
        arr = Image.open(f)
        arr = arr.resize((IMG_HEIGHT,IMG_WIDTH), Image.ANTIALIAS)
        arr = np.reshape(arr, (-1, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS))
        if img_array is None:
            img_array = arr
        else:
            img_array = np.vstack((img_array, arr))        
    queue.put(img_array)
#%%
def dump_array(fname,arr):
    with open(fname,'wb') as f:
        pickle.dump(arr,f)
#%%
def load_pickled_array(fname,arr):
    with open(fname, 'rb') as f:
        return pickle.load(f)
#%%
# using threading combine training array and labels for training data
def get_training_data():
    threads_list = list()
    train_x = None
    train_y = []
    queue = initialize_queue()
    # iterate over num of threads to create
    for thread_index in range(num_train_threads):
        start_index = thread_index * batch_size
        end_index = (thread_index + 1) * batch_size
        file_batch = train_imgs[start_index:end_index]
        thread = Thread(target =get_img_array_labels, args=(file_batch, queue))
        thread.start()
        print("Thread: {}, start index: {}, end index: {}".format(thread.name, start_index, end_index))
        threads_list.append(thread)
    
    # join threads
    for t in threads_list:
        t.join()
    while not queue.empty():
        arr, labels = queue.get()
        train_y.extend(labels)
        if train_x is None:
            train_x = arr
        else:
            train_x = np.vstack((train_x, arr))
    return train_x, train_y
#%%
# using multithreading combine testing array for testing data
def get_testing_data():
    threads_list = list()
    test_x = None
    queue = initialize_queue()
    # iterate over num of threads to create
    for thread_index in range(num_test_threads):
        start_index = thread_index * batch_size
        end_index = (thread_index + 1) * batch_size
        file_batch = train_imgs[start_index:end_index]
        thread = Thread(target =get_img_array, args=(file_batch, queue))
        thread.start()
        print("Thread: {}, start index: {}, end index: {}".format(thread.name, start_index, end_index))
        threads_list.append(thread)
    
    # join threads
    for t in threads_list:
        t.join()
        print("Thread: {} joined", t.name)
    while not queue.empty():
        arr= queue.get()
        if test_x is None:
            test_x = arr
        else:
            test_x = np.vstack((test_x, arr))
    return test_x
#%%
train_x, train_y = get_training_data()
#%%
print(train_x.shape)
print(len(train_y))
#%%
test_x =get_testing_data()
print(test_x.shape)
#%%
import pickle
dump_array('train_arr.pickle',train_x)
dump_array('train_labels.pickle',train_y)
#%%
# dump testing data
dump_array('test_arr.pickle',test_x)
#%%
print("train_x shape",train_x.shape)
print("test_x shape", test_x.shape)
# convert train_y to np. array
train_y = np.array(train_y)
print("train_y.shape", train_y.shape)
#%%
# mean normalize train and test images
train_x = train_x/255
test_x = test_x/255
#%%
# import required packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils, to_categorical

from sklearn.model_selection import train_test_split

#%%
# CNN model
# CNN model
model = Sequential()

# -----------------------------------------------------------------------------------
# conv 1
model.add(Conv2D(16, (3,3), input_shape=(50,50,3))) # 148,148,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# max pool 1
model.add(MaxPooling2D(pool_size=(2,2),strides=2))          # 72,72,32

# -----------------------------------------------------------------------------------
# conv 2
model.add(Conv2D(16, (3,3)))                      # 68,68,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.5))

# max pool 2
model.add(MaxPooling2D(pool_size=(2,2),strides=2))          # 34,34,32
# -----------------------------------------------------------------------------------

# conv 3
model.add(Conv2D(32, (3,3)))                      # 32,32,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))

# max pool 3
model.add(MaxPooling2D(pool_size=(2,2),strides=2))          # 17,17,32
# -----------------------------------------------------------------------------------

# conv 4
model.add(Conv2D(32, (3,3)))                      # 15,15,32
model.add(BatchNormalization(axis=3))
model.add(Activation('relu'))
#model.add(Dropout(0.7))
# max pool 4
model.add(MaxPooling2D(pool_size=(2,2),strides=2))  # 7,7,32

# flatten
model.add(Flatten())


# fc layer 1
model.add(Dense(512, activation='relu'))

#model.add(Dropout(0.7))

#model.add(Dense(256, activation='relu'))

#model.add(Dropout(0.5))

# fc layer 2
model.add(Dense(2, activation='softmax'))

#%%
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#%%
predictions = model.predict(test_x, batch_size=32, verbose=1)
#%%


model.summary()


#%%
import matplotlib.pyplot as plt
#matplotlib inline
fig=plt.figure()

for index in range(12):
    # cat: [1,0]
    # dog: [0,1]
    y = fig.add_subplot(3,4,index+1)
    #model_out = model.predict([data])[0]
    img = test_x[index]
    model_out = predictions[index]
    if np.argmax(model_out) == 0: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(img)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()

#%%

with open('submission.csv','w') as f:
    f.write('id,label\n')
    for index in range(len(test_imgs)):
        img_id =basename(test_imgs[index]).split(".")[0]
        prob = (predictions[index,0])
        #print("index: {}, img_id: {}, prob:{}".format(index,img_id, prob))
        f.write("{},{}\n".format(img_id, prob))
