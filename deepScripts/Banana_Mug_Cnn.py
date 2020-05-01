import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import os
import subprocess
import numpy as np
import cv2 as cv


def main():

    shuffledImages, shuffledLabels = shuffleData(images, labels, shapes) #Shuffle images and labels 
    print(shuffledLabels.shape)

    #Split data into train and test
    trainImages, testImages, trainLabels, testLabels = splitData(shuffledImages, shuffledLabels, SPLIT_PERCENTAGE)

    print("Creating Model")
    #Create model with pre-processing and classifier layers
    model = models.Sequential()
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (None, None, None)))
    model.add(layers.ZeroPadding2D((1, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.ZeroPadding2D((1,1)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

    #Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(2))

    print("Compiling Model")
    model.compile(optimizer = 'adam',
                loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                metrics = ['categorical_accuracy'])
    
    history = model.fit(trainImages, trainLabels, epochs = 12)

    model.save("BananaMugDense.h5")

    #newModel = tf.keras.models.load_model("BananaMugDense.h5")
    testLoss, testAcc = model.evaluate(testImages, testLabels, verbose = 2)
    print(testAcc)

    print("Done")


def splitData(shuffledImages, shuffledLabels, SPLIT_PERCENTAGE):
    if SPLIT_PERCENTAGE >= .97:
        
        return trainImages, trainLabels

    trainImages = shuffledImages[0 : int(SPLIT_PERCENTAGE * shuffledImages.shape[0])]
    testImages = shuffledImages[int(SPLIT_PERCENTAGE * shuffledImages.shape[0]): ]
    trainLabels = shuffledLabels[0 : int(SPLIT_PERCENTAGE * shuffledImages.shape[0])]
    testLabels = shuffledLabels[int(SPLIT_PERCENTAGE * shuffledImages.shape[0]) : ]

    print("dsjkf" , trainImages.shape, trainLabels.shape)
    

    return trainImages, testImages, trainLabels, testLabels

"""
@description Shuffles Images and labels with respect to each other
"""
def shuffleData(images, labels, shape):
    tf.compat.v1.disable_eager_execution()

    x = tf.compat.v1.placeholder(tf.float32, shape = (None, None, None, None))
    y = tf.compat.v1.placeholder(tf.uint32, (None))

    indices = tf.range(start = 0, limit = tf.shape(x)[0], dtype = tf.int32)
    shuffledIndices = tf.random.shuffle(indices)

    shuffledX = tf.gather(x, shuffledIndices)
    shuffledY = tf.gather(y, shuffledIndices)

    with tf.compat.v1.Session() as sess:
        x_res, y_res = sess.run([shuffledX, shuffledY],
                                feed_dict = {x: images, y: labels})

    
    return x_res, y_res
def resizeSingleImage(image, scalePercent):
    newHeight = int(image.shape[0] * scalePercent / 100)
    newWidth = int(image.shape[1] * scalePercent / 100)
    newDim = (newWidth, newHeight)

    resizedImg = cv.resize(image, newDim, cv.INTER_AREA)

    return resizedImg

def makeImagesAndLabels(anomoly, skip, folder):
    os.chdir(folder)

    y = subprocess.Popen(["ls"], shell=True, stdout=subprocess.PIPE) #List all things in current directory
    x = str(y.communicate())
    imageNames = x.split("\\n") #Holds all image names
    
    height , width, colAmount = resizeSingleImage(cv.imread(imageNames[7]), SCALE_IMAGE_TO).shape
    imageNames[0] = imageNames[0][imageNames[0].index(anomoly) : len(imageNames[0])] #Corrects string split anomaly
    imageNames1 = np.asarray(imageNames)

    imageAmount = imageNames1.shape[0] - 1 
    images = np.empty((imageAmount, NEW_HEIGHT, NEW_WIDTH, colAmount), dtype = np.uint8)
    labels = np.empty(imageNames1.shape, dtype = np.uint8)


    for x in range(imageNames1.shape[0]):

        if(x < imageNames1.shape[0] - 1): #Reads images into array
            imagePath = imageNames1[x]
            image = cv.imread(imagePath)
            image = resizedImg = cv.resize(image, (NEW_HEIGHT, NEW_WIDTH), cv.INTER_AREA)
            images[x] = image

        if skip == True:
            labels[x] = 1
            continue

        if x < 535: #Creates labels array for each image
            labels[x] = 0   #banana
        elif x >=535:
            labels[x] = 1 # mug

    images = images / 255.0
    os.chdir(origDir)
    return images, labels, (height, width, colAmount)

if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    #sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    SPLIT_PERCENTAGE = .80
    SCALE_IMAGE_TO = 100
    origDir = os.getcwd()
    print("starting")

    NEW_HEIGHT, NEW_WIDTH = (256, 256)
    images, labels, shapes = makeImagesAndLabels('ban', False, 'images') #Create labels and images array
    #extraImages, extraLabels, extraShapes = makeImagesAndLabels('out', True, 'testImages')

    main()