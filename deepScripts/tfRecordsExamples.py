import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import subprocess
import numpy as np
import cv2 as cv

def main():

    shuffledImages, shuffledLabels = shuffleData() #Shuffle images and labels 
    print(shuffledLabels.shape)
    #print(shuffledLabels[34])
    #while cv.waitKey(1) <0:
    #    cv.imshow("picture", shuffledImages[34])

    #Split data into train and test
    trainImages = shuffledImages[0 : int(SPLIT_PERCENTAGE * shuffledImages.shape[0])]
    testImages = shuffledImages[int(SPLIT_PERCENTAGE * shuffledImages.shape[0]): ]
    trainLabels = shuffledLabels[0 : int(SPLIT_PERCENTAGE * shuffledLabels.shape[0])]
    testLabels = shuffledLabels[int(SPLIT_PERCENTAGE * shuffledLabels.shape[0]) : ]

    #Create TF.dataRecord for train and test
    createDataRecord('train.tfrecords', trainImages, trainLabels)
    createDataRecord('test.tfrecords', testImages, testLabels)
    
    filenames = ['train.tfrecords'] #, 'test.tfrecords']
    dataset = tf.data.TFRecordDataset(filenames = filenames)
    print(dataset)
    #Create model with pre-processing and classifier layers


    print("Done")


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


"""
@description Writes data to .tfRecords file
"""
def createDataRecord(fileOutName, imagesArray, labelsArray):
    writer = tf.compat.v1.python_io.TFRecordWriter(fileOutName) #Open tfrecords file

    for i in range(imagesArray.shape[0]):
        img = imagesArray[i]
        label = labelsArray[i]

        if img is None: #Skip iteration if img is null
            continue

        feature = {
            'image_raw': _bytes_feature(img.tostring()),
            'label': _int64_feature(label)
        }
        example = tf.train.Example(features = tf.train.Features(feature = feature))

        writer.write(example.SerializeToString())

    writer.close()


"""
@description Shuffles Images and labels with respect to each other
"""
def shuffleData():
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.uint8, shape = (None, height, width, colAmount))
    y = tf.compat.v1.placeholder(tf.uint32, (None))

    indices = tf.range(start = 0, limit = tf.shape(x)[0], dtype = tf.int32)
    shuffledIndices = tf.random.shuffle(indices)

    shuffledX = tf.gather(x, shuffledIndices)
    shuffledY = tf.gather(y, shuffledIndices)

    with tf.compat.v1.Session() as sess:
        x_res, y_res = sess.run([shuffledX, shuffledY],
                                feed_dict = {x: images, y: labels})

    
    return x_res, y_res

def makeImagesAndLabels():
    os.chdir('images')

    y = subprocess.Popen(["ls"], shell=True, stdout=subprocess.PIPE) #List all things in current directory
    x = str(y.communicate())
    imageNames = x.split("\\n") #Holds all image names

    imageNames[0] = imageNames[0][imageNames[0].index('ban') : len(imageNames[0])] #Corrects string split anomaly
    imageNames1 = np.asarray(imageNames)

    imageAmount = imageNames1.shape[0] - 1 
    images = np.empty((imageAmount, height, width, colAmount), dtype = np.uint8)
    
    labels = np.empty(imageNames1.shape, dtype = np.uint8)
    for x in range(imageNames1.shape[0]):

        if(x < imageNames1.shape[0] - 1): #Reads images into array
            imagePath = imageNames1[x]
            image = cv.imread(imagePath)
            images[x] = image

        if x < 535: #Creates labels array for each image
            labels[x] = 0   #banana
        elif x >=535:
            labels[x] = 1 #mug

    os.chdir(origDir)
    return images, labels

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    origDir = os.getcwd()
    height ,width, colAmount = cv.imread('images/banana001.jpg').shape 

    images, labels  = makeImagesAndLabels() #Create labels and images array

    SPLIT_PERCENTAGE = .75
    
    main()