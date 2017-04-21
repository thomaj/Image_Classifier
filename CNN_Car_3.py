# Different arcitechure and format compared to the other two Conv nets
# Used layout and template from https://github.com/nlintz/TensorFlow-Tutorials/blob/master/05_convolutional_net.py

#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from PIL import Image
import random
import os
import csv
from tensorflow.examples.tutorials.mnist import input_data

FILES_AND_CLASSES_PATH = os.path.abspath('FileNameAndClasses.csv')
IMAGES_DIR = os.path.abspath('Images_Dodge')
SAVE_PATH = os.path.abspath('car_cnn_model_3_3')

batch_size = 120
test_size = 630


def load_dataset(files_and_classes, images_dir):
    """Returns the needed arrays or training/testing images and labels"""
    data = {
        'train': {'images': None, 'labels': None},
        'test': {'images': None, 'labels': None}
    }

    print ('Starting import of data')

    # Import the csv file containing the image file names and class numbers
    with open(files_and_classes, 'rb') as f:
        reader = csv.reader(f)
        fnameClass = list(reader)

    # Get size of the training/evaluation sets
    num_images = len(fnameClass)
    num_training = int(num_images * 0.8)

    training_imgs = []
    training_labels = []
    print (num_training)
    for i in range(num_training):
        elem = random.choice(fnameClass)    # Get random element
        image = Image.open(os.path.join(images_dir, elem[0]))
        img_reshaped = np.reshape(image.getdata(), (128, 128, 3))
        training_imgs.append(img_reshaped)
        image.close()   # Close the open image file
        training_labels.append(elem[1])
        fnameClass.remove(elem)  # Remove this image from being picked again

    # Set the testing data to what is left
    testing_imgs = []
    testing_labels = []
    for elem in fnameClass:
        image = Image.open(os.path.join(images_dir, elem[0]))
        img_reshaped = np.reshape(image.getdata(), (128, 128, 3))
        testing_imgs.append(img_reshaped)
        image.close()   # Close the open image file
        testing_labels.append(elem[1])

    # Set the dictionary so is has this data
    data['train']['images'] = np.asarray(training_imgs, dtype=np.float32)
    data['train']['labels'] = np.asarray(training_labels, dtype=np.int32)
    data['test']['images'] = np.asarray(testing_imgs, dtype=np.float32)
    data['test']['labels'] = np.asarray(testing_labels, dtype=np.int32)

    print(data['test']['images'].shape)

    print ('Finished importing data')

    return data


def make_one_hot(reg_data):
    #print reg_data.shape
    one_hot = np.array([[0 for i in range(15)] for x in range(len(reg_data))])
    for i, label in zip(range(len(reg_data)), reg_data):
        #print label
        one_hot[i][int(label)] = 1

    return one_hot


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,           # l1a shape=(?, 128, 128, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],   # l1 shape=(?, 64, 64, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,          # l2a shape=(?, 64, 64, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],   # l2 shape=(?, 32, 32, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,          # l3a shape=(?, 32, 32, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],   # l3 shape=(?, 16, 16, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  # reshape to (?, 32768)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# Load the training and eval data
loaded_data = load_dataset(FILES_AND_CLASSES_PATH, IMAGES_DIR)
trX = loaded_data['train']['images']
trY = make_one_hot(loaded_data['train']['labels'])
teX = loaded_data['test']['images']
teY = loaded_data['test']['labels']
trX = trX.reshape(-1, 128, 128, 3)  # 128x128x3 input img
teX = teX.reshape(-1, 128, 128, 3)  # 128x128x3 input img

X = tf.placeholder("float", [None, 128, 128, 3])
Y = tf.placeholder("float", [None, 15])

w = init_weights([3, 3, 3, 32])       # 3x3x3 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x64 conv, 128 outputs
w4 = init_weights([128 * 16 * 16, 625]) # FC 128 * 16 * 16 inputs, 625 outputs
w_o = init_weights([625, 15])         # FC 625 inputs, 15 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    init_op.run()

    bestAccuracySoFar = np.float64(-0.1)
    for i in range(64):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                    p_keep_conv: 0.8, p_keep_hidden: 0.9})

        if i % 2 == 0:
            test_indices = np.arange(len(teX)) # Get A Test Batch
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]

            pr = sess.run(predict_op, feed_dict={X: teX[test_indices],
                p_keep_conv: 1.0, p_keep_hidden: 1.0})
            # print(pr)
            # print(teY[test_indices])
            accuracy = np.mean(teY[test_indices] == pr)
            print(i, accuracy)

            # Only save if the accuracy is greater than best so far
            if (accuracy-bestAccuracySoFar).all():
                bestAccuracySoFar = pr
                saver.save(sess, SAVE_PATH)

