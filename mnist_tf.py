#importing the required dependencies
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.utils import np_utils

#load the data
data = pd.read_csv('digits.csv')

#take all the labels separately
label = data['label']
label = np.asarray(label)
label = np.reshape(label, [-1, 1])

#convert labels to one-hot encoded values
label = np_utils.to_categorical(label)

#separate images
image = data.iloc[0:, 1:]
image = np.asarray(image)
image = np.reshape(image, [-1, 28, 28, 1])

#define placeholders
x_ = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ = tf.placeholder(tf.float32, [None, 10])

#Helper functions defination
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, -2, 2))

def create_bias(shape):
    return tf.Variable(tf.constant(0.01))

def conv(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(x):
    return tf.nn.relu(x)

def flatten(x):
    flag = x.get_shape()
    num_elements = flag[1:4].num_elements()
    reshape = tf.reshape(x, [-1, num_elements])
    return reshape

def fully_conn(x, w, b):
    layer = tf.matmul(x, w) + b
    return tf.nn.relu(layer)

#convolutional layer

w1 = create_weights([5, 5, 1, 32])
conv1 = conv(x_, w1)
conv_layer1 = conv_layer(conv1)
max1 = maxpool(conv_layer1)
w2 = create_weights([5, 5, 32, 64])
conv2 = conv(max1, w2)
conv_layer2 = conv_layer(conv2)
max2 = maxpool(conv_layer2)
w3 = create_weights([3, 3, 64, 128])
conv3 = conv(max2, w3)
conv_layer3 = conv_layer(conv3)
max3 = maxpool(conv_layer3)
w4 = create_weights([3, 3, 128, 256])
conv4 = conv(max3, w4)
conv_layer4 = conv_layer(conv4)
max4 = maxpool(conv_layer4)

#flatten layer
flat1 = flatten(max4)

#fully connected layer
w_fc1 = create_weights([flat1.get_shape()[1:4].num_elements(), 1024])
b_fc1 = create_bias([1024])
fc1 = fully_conn(flat1, w_fc1, b_fc1)
w_fc2 = create_weights([1024, 128])
b_fc2 = create_bias([128])
fc2 = fully_conn(fc1, w_fc2, b_fc2)
w_fc3 = create_weights([128, 16])
b_fc3 = create_bias([16])
fc3 = fully_conn(fc2, w_fc3, b_fc3)
w_fc4 = create_weights([16, 10])
b_fc4 = create_bias([10])
fc4 = fully_conn(fc3, w_fc4, b_fc4)

y = fc4

#defining cost
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
cost = tf.reduce_mean(cross_entropy)

#using adam optimizer to reduce cost
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)

#accuracy
y_pred_cls = tf.argmax(y, dimension=1)
y_true = tf.argmax(y_, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#running tf Session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(10):
        sess.run(optimizer, feed_dict = {x_: image[:40000], y_: label[:40000]})
        acc = sess.run(accuracy, feed_dict = {x_: image[40000:], y_: label[40000:]})
        print(i, acc)