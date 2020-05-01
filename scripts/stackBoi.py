#!/usr/bin/env python

import numpy as np
import cv2 as cv
import torch
import os
import subprocess
import stackBoiMethods_cy #Accelerated methods with c implementation

"""
@Description This program stacks images on top of each other then
        determines the median of each pixel and outputs a single smooth denoised image
"""
def main():
    print("Reading Images...")
    images = readImages()#Read in images

    #Stack Images
    blues, greens, reds = stackBoiMethods_cy.stackImages(images, cuda)
    print(blues) 
    #Medianize Images
    print("Medianizing Images...")
    finalImage = stackBoiMethods_cy.medianize(blues, greens, reds, images)#Apply median filter to images
    
    print("Saving Image...")
    cv.imwrite('finalImag.png', finalImage)
    #Display Image
    print("Displaying Image...")
    while cv.waitKey(1) < 0:
        cv.imshow('Final', finalImage)

    print("Done")

def resizeSingleImage(image, scalePercent):
    newHeight = int(image.shape[0] * scalePercent / 100)
    newWidth = int(image.shape[1] * scalePercent / 100)
    newDim = (newWidth, newHeight)

    image = image.cpu().numpy()
    resizedImg = cv.resize(image, newDim, cv.INTER_AREA)

    cudaImg = torch.from_numpy(resizedImg)
    cudaImg = cudaImg.to(cuda)
    return cudaImg
"""
@description Resize an array of images
@returns resized images
"""
def resizeImages(images, scalePercent):
    newHeight = int(height* scalePercent / 100)
    newWidth = int(width * scalePercent / 100)
    newDim = (newWidth, newHeight)
    print("Resized Image Dimensions: %i %i %i " %(newHeight, newWidth, colAmount))

    images = images.cpu().numpy() #Convert images to numpy array for cv
    resizedImages = np.empty((images.shape[0], newHeight, newWidth, colAmount), dtype = np.uint8)

    for x in range(images.shape[0]):
        resizedImg = cv.resize(images[x], newDim, cv.INTER_AREA)
        resizedImages[x] = resizedImg

    resizedCuda = torch.from_numpy(resizedImages)
    resizedCuda = resizedCuda.to(cuda)
    return resizedCuda


"""
@Description Sharpens an array filled with images
@returns Sharpened images
"""
def sharpenImages(images, *args):
    images = images.cpu().numpy()
    print(images.dtype)
    sharpBoi = np.zeros_like(images, dtype = np.uint8) #Holds single image of BGRs
    
    print("Sharpedn: " ,images.shape)
    if(args[0] != 0 and args[0] == 1): #For single images
        for i in range(3):
            sharpBoi[:, :, i] = laplacian(images[:, :, i],laplacianStrength) #Sharpen Image
        return sharpBoi

    sharpBois = np.zeros_like(images) #Holds all images
    for x in range(images.shape[0]): #Loop through all images
        for i in range(3): #Loop through each color
            sharpBoi[:, :, i] = laplacian(images[x, :, :, i], laplacianStrength) #Sharpen Images
        sharpBois[x] = sharpBoi #Store Images in array

    return sharpBois

"""
@description Sharpens image 
@returns Sharpened image
"""
def laplacian(image, strength):
    gaussian = cv.GaussianBlur(image, (3, 3), 0) #Apply noise filter

    laplacian = cv.Laplacian(gaussian, cv.CV_32F) #Apply edge detector

    sharp = image - (strength * laplacian) #Sharpen the original image by subtracting the negative edges
    sharp[sharp > 255] = 255 #Set cap of pixels to 255
    sharp[sharp < 0] = 0 #Set lower-limit cap to 0

    return sharp



"""
@description Stacks the images values into blue green and red images
@returns Arrays for blues, greens, reds containing all the value from each picture
"""
def stackImages(images):
    #Make arrays to store stacked images for RGB
    blues = torch.empty([images.shape[1], images.shape[2], images.shape[0]], dtype = torch.uint8, device = cuda)
    greens = torch.empty([images.shape[1], images.shape[2], images.shape[0]], dtype = torch.uint8, device = cuda)
    reds = torch.empty([images.shape[1], images.shape[2], images.shape[0]], dtype = torch.uint8, device = cuda) 

    print(images.dtype)
    print("Images: ", images.shape[0])
    resizedHeight = images.shape[1]
    resizedWidth = images.shape[2]
    for x in range(images.shape[0]): #loop through all the images
        print("Stacking image %i" %(x))
        for h in range(resizedHeight): #Loops through the all pixels in the images
            for w in range(resizedWidth):
                blues[h][w][x] = images[x][h][w][0]
                greens[h][w][x] = images[x][h][w][1]
                reds[h][w][x] = images[x][h][w][2]

    return blues, greens, reds

"""
@description: Moves all images in a folder to tensor array
@returns array holding all images in np for
"""
def readImages():
    os.chdir("../stackedImages") #Moves to dir with images

    y = subprocess.Popen(["ls"], shell=True, stdout=subprocess.PIPE) #List all things in current directory
    x = str(y.communicate())
    imageNames = x.split("\\n") #Holds all image names

    allImages = np.empty((len(imageNames), height, width, colAmount), dtype = np.uint8) #Holder for images 
    
    imageNames[0] = imageNames[0][imageNames[0].index('IMG') : len(imageNames[0])] #Corrects string split anomaly
    for x in range(len(imageNames) - 1): #Loops through names of images
        image = cv.imread(imageNames[x])
        allImages[x] = image #appends image

    #Moves all images to tensor on cuda
    allImages = torch.from_numpy(allImages)
    allImages = allImages.to(cuda)

    os.chdir(origDir) #Moves back to original directory
    return allImages


if __name__ == '__main__':
    origDir = os.getcwd()
    height ,width, colAmount = cv.imread('../stackedImages/IMG_20200301_181534.jpg').shape 
    newHeight = newWidth = 0 

    print("Original Image Dimensions: %i %i %i" %(height, width, colAmount))
    cv.startWindowThread()
    cv.namedWindow('Final', cv.WINDOW_NORMAL)
    cv.resizeWindow('Final', 600, 600)

    cuda = torch.device('cuda:0')
    laplacianStrength = 1.1
    
    main()