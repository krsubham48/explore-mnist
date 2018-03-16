#importing the required dependencies

import numpy as np
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

#loading the data file
data = pd.read_csv('digits.csv')

#separating images and labels from the data file
label = data['label']
image = data.iloc[0:, 1:]

#converting the images and labels as numpy array from pandas dataframe, for furthur computations
label = np.asarray(label)
image = np.asarray(image)
label = np.reshape(label, [-1, 1])
image = np.reshape(image, [-1, 28, 28, 1])

#converting labels to one hot encodings
label = np_utils.to_categorical(label)

#applying the keras sequential model to fit the data
model = Sequential()
model.add(Conv2D(16, (5, 5), input_shape=(28, 28, 1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

epochs = 20
lrate = 0.01
sgd = SGD(lr=lrate)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(image[:40000], label[:40000], epochs=epochs, batch_size=128)

scores = model.evaluate(image[40000:], label[40000:])
print(scores)