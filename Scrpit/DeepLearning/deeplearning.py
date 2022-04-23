# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
#%% Import modules
import keras
keras.__version__
import matplotlib.pyplot as plt
import numpy as np

#%% Listing 5.1 Instatiating a small convnet

from keras import layers
from keras import models

model = models.Sequential() # Uma cada atras da outra
model.add(layers.Conv2D(32, (3,3), # primeira camada convolucional, com 32 neuronios
                        activation = 'relu', # Adiciona não linearidade a convolução da rede
                        input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.summary()
# A não-linearidade ajuda a rede a ter classificadores mais robustos, que não
# se limitam a fazer retas para classificar

#%% Listing 5.2 Adding a classifier on top of the convnet

model.add(layers.Flatten()) # Vetorização
model.add(layers.Dense(64, activation='relu')) 
model.add(layers.Dense(10, activation='softmax')) # Pra normalizar
model.summary()

#%% Listing 5.3 Training the convnet on MNIST images

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
idx = 10
plt.imshow(255 - train_images[idx], cmap = 'gray')
plt.xlabel('Class = ' + str(train_labels[idx]))
plt.show()

#%% 

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer = 'rmsprop', 
              loss='categorical_crossentropy', # METRICA DE PERDA
              metrics=['accuracy']) # METRICA DE QUALIDADE
#%% Train and save model
model.fit(train_images, train_labels, epochs=5, batch_size=64)
model.save('model-5-1.h5')
from keras.models import load_model
model = load_model('model-5-1.h5')
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc # 99.08
#%%

pred_labels = model.predict_classes(test_images)
pred_scores = model.predict(test_images)
labels = np.argmax(test_labels, axis = 1)
idxs = ~(pred_labels == labels)
ind = np.where(idxs)[0]
for i in ind:
    img = test_images[i]
    plt.imshow(255 - img[:, :, 0], cmap='gray')
    plt.xlabel('Class = ' + str(labels[i]) + ' pred.class = ' + str(pred_labels[i]))
plt.show()
    