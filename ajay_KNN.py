# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:02:57 2017

@author: Mike James
"""



import numpy as np

import operator

from os import listdir

import time

import pandas as pd

import matplotlib.pyplot as plt

 

#Harrington, Peter. Machine Learning in Action. Manning, 2012.

#Utilizes Euclidian Distance to make prediction

def make_prediction(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()    
    classCount={}         

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

 

  

#Harrington, Peter. Machine Learning in Action. Manning, 2012.

#Convert the 32x32 image of an digit between 0-9 into a vector

def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

#Return a list of files for testing/training

def get_image_files(path):
    return listdir(path)

#Make graph of data to discover ideal value of k

def make_graph(data, data_len):
    kNNdf = pd.DataFrame(np.array(data).reshape(int(data_len / 5), 5), columns = ["k","NumberOfErrors","ErrorRate","CorrectRate", "ElapsedTime"])

    if(len(kNNdf) > 1):
        plt.plot(kNNdf['k'], kNNdf['CorrectRate'])
        plt.ylabel('Correctly Classified')
        plt.xlabel('k')
        plt.show()   

        plt.plot(kNNdf['k'], kNNdf['ErrorRate'])
        plt.ylabel('Incorrectly Classified')
        plt.xlabel('k')
        plt.show()    

        plt.plot(kNNdf['k'], kNNdf['NumberOfErrors'])
        plt.ylabel('Number of errors')
        plt.xlabel('k')
        plt.show()   

        plt.plot(kNNdf['k'], kNNdf['ElapsedTime'])
        plt.ylabel('Time (seconds)')
        plt.xlabel('k')
        plt.show()   

    return

#Initiate the classifying process by first reading training files and making a prediction with the test data

def perform_handwritten_test(k):

    try:

        handwritten_labels = []       
        training_image_files = get_image_files('DigitsFolder/TrainingDigits')       
        training_number_of_files = len(training_image_files)       
        training_array = np.zeros((training_number_of_files, 1024))  #Initialize with 0s       
        for i in range(training_number_of_files):
            training_filename = training_image_files[i]
            training_filename_wo_ext = training_filename.split('.')[0]     #Remove file extension .txt
            handwritten_labels.append(int(training_filename_wo_ext.split('_')[0]))
            training_array[i,:] = img2vector('DigitsFolder/TrainingDigits/%s' % training_filename)       

        
        testing_image_files = get_image_files('DigitsFolder/TestingDigits')        #Get testing data       
        error_count = 0.0
        testing_number_of_files = len(testing_image_files)   
        for i in range(testing_number_of_files):
            testing_filename = testing_image_files[i]
            testing_filename_wo_ext = testing_filename.split('.')[0]     #Remove file extension .txt
            class_number = int(testing_filename_wo_ext.split('_')[0])
            test_vector = img2vector('DigitsFolder/TestingDigits/%s' % testing_filename)
            result = make_prediction(test_vector, training_array, handwritten_labels, k) #1,2,3,etc = k
            print ("Predicted: %d, Actual: %d, Instance Count: %d, k = %d" % (result, class_number, i, k))

            if (result != class_number):

                error_count += 1.0     

        print ("\nTotal number of errors are: %d" % error_count)

        error_rate = error_count/float(testing_number_of_files)
        correct_rate = (abs(1 - error_rate))

        print ("\nTotal error rate is: %f" % error_rate)

        print ("\nTotal correct rate is %f" % correct_rate) 

        return error_count, error_rate, correct_rate   

    except Exception as e:

        print(e)

       

#Start Classifier

def begin_classify(k):

    result_set = []
    for i in range(k): #k = 1, 2, 3, etc.
        start_time = time.time()
        error_count, error_rate, correct_rate = perform_handwritten_test(i+1)
        elapsed_time = round((time.time() - start_time),2)
        print("Execution time: %s seconds " % elapsed_time)
        
        result_set.append(i)
        result_set.append(error_count)
        result_set.append(error_rate)
        result_set.append(correct_rate)
        result_set.append(elapsed_time)

    make_graph(result_set, len(result_set))

    return

 

#Main menu

def main_menu():

    print ("kNN - Nearest Neighbor Handwritten Digit Classifier\n")

    try:
        choice = input("Enter a value for k (1-50):   ")
    except EOFError:
        print("EOFerror")
    except KeyboardInterrupt:
        print("Operation cancelled")
    else:
        if choice >= '1' and choice <= '50':
            begin_classify(int(choice))

    return

   

if __name__ == "__main__":
    main_menu()

   
