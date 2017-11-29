# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:43:03 2017

@author: Mike James
"""



import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
# 
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def get_training_file():
    handwritten_labels = []  
    training_image_files = get_image_files('DigitsFolder/TrainingDigits')       
    training_number_of_files = len(training_image_files)       
    training_array = np.zeros((training_number_of_files, 1024)) 
  
    for i in range(1):
        training_filename = training_image_files[i]
        training_filename_wo_ext = training_filename.split('.')[0]     #Remove file extension .txt
        handwritten_labels.append(int(training_filename_wo_ext.split('_')[0]))
        training_array[i,:] = img2vector('DigitsFolder/TrainingDigits/%s' % training_filename) 
    return training_array            

def get_training_labels():
    handwritten_labels = []  
    training_image_files = get_image_files('DigitsFolder/TrainingDigits')       
    training_number_of_files = len(training_image_files)       
  
    for i in range(1):
        training_filename = training_image_files[i]
        training_filename_wo_ext = training_filename.split('.')[0]     #Remove file extension .txt
        handwritten_labels.append(int(training_filename_wo_ext.split('_')[0]))
    
    return handwritten_labels           
    
        
def get_test_images():     
    testing_image_files = get_image_files('DigitsFolder/TestingDigits')        #Get testing data       
#    error_count = 0.0
#    testing_number_of_files = len(testing_image_files)   
    for i in range(1):
        testing_filename = testing_image_files[i]
#        testing_filename_wo_ext = testing_filename.split('.')[0]     #Remove file extension .txt
#        class_number = int(testing_filename_wo_ext.split('_')[0])
        test_vector = img2vector('DigitsFolder/TestingDigits/%s' % testing_filename)
    return test_vector


def get_test_labels():
    handwritten_labels = []  
    training_image_files = get_image_files('DigitsFolder/TestingDigits')       
    training_number_of_files = len(training_image_files)       
  
    for i in range(1):
        training_filename = training_image_files[i]
        training_filename_wo_ext = training_filename.split('.')[0]     #Remove file extension .txt
        handwritten_labels.append(int(training_filename_wo_ext.split('_')[0]))
    
    return handwritten_labels          


# Softmax regression

x = tf.placeholder(tf.float32, [None, 1024])
W = tf.Variable(tf.zeros([1024, 10]))
b = tf.Variable(tf.zeros([10]))  

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()


for _ in range(1000):
    train_x = get_training_file()
    train_y = get_training_labels()
    sess.run(train_step, feed_dict={x: train_x, y_:train_y})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

test_images = get_test_images()
test_labels = get_test_labels()
print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))

     