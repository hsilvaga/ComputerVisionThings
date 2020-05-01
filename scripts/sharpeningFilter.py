#!/usr/bin/env python
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy as cpy
#import CV libraries

kernel = np.empty((3,3))
#convolutionModifier = 1.0

def main():
    print("Fixing RGB matrix sizes...")
    fixedRedMat = reshapeToKernelSize(redMat, 0)
    fixedGreenMat = reshapeToKernelSize(greenMat, 1)
    fixedBlueMat = reshapeToKernelSize(blueMat, 2)

    #Create smoothed red, green, and blue images with convolution
    print("Smoothing RGB images...")
    kernel.fill(1)
    smoothedRedImage = createPixelArray(fixedRedMat, 9)
    smoothedGreenImage = createPixelArray(fixedGreenMat, 9)
    smoothedBlueImage = createPixelArray(fixedBlueMat, 9)

    print("Increasing brightness of RGB images...")
    #Create 2x image for original R, G, B with covolution
    kernel.fill(0)
    kernel[1][1] = 2
    twoXRed = createPixelArray(fixedRedMat, 1)
    twoXBlue = createPixelArray(fixedBlueMat, 1)
    twoXGreen = createPixelArray(fixedGreenMat, 1)

    #Create sharpened image by subtracting the twoX by the smoothedImage
    print("Sharpening RGB images...")
    sharpenedRed = np.subtract(twoXRed, smoothedRedImage)
    sharpenedGreen = np.subtract(twoXGreen, smoothedGreenImage)
    sharpenedBlue = np.subtract(twoXBlue, smoothedBlueImage)

    #Merge the images back to single matrix
    print("Merging sharpened RGB images to single picture...")
    sharpenedRGBImg = mergedImages(sharpenedRed, sharpenedGreen, sharpenedBlue)

    print "Done"
    plt.imshow(sharpenedRGBImg, vmin=0, vmax=255)
    plt.show()

#Merges red, green, and blue images to one matrix
def mergedImages(redImage, greenImage, blueImage):
    RGBImage = np.empty((len(redImage), len(redImage[0]), 3))

    for r in range(len(redImage)):
        for c in range(len(redImage[0])):
            for h in range(3):
                if h == 0:
                    RGBImage[r][c][h] = redImage[r][c] #index (x, y, 0)
                elif h == 1:
                    RGBImage[r][c][h] = greenImage[r][c] #index (x, y, 1)
                elif h == 2:
                    RGBImage[r][c][h] = blueImage[r][c] #index at (x, y, 2)
    print("Image size: (%d, %d)", RGBImage.shape[0], RGBImage.shape[1])
    return RGBImage

def createPixelArray(fixedOriginal,convoltionModifier):
    #fixedOriginal = reshapeToKernelSize()
    currentPixel = (0,0)
    newImage = np.empty((len(fixedOriginal), len(fixedOriginal[0])))
    newImage.fill(0)

    for c in range(1, len(fixedOriginal) -1):
        for r in range(1, len(fixedOriginal[0]) - 1) :
            currentPixel =(c,r)
            convolutedNumber = convolution(fixedOriginal, currentPixel, convoltionModifier)
            newImage[c-1][r-1] = convolutedNumber

    return newImage

def convolution(fixedOriginal, currentPixel, convolutionModifier):
    global kernel

    c,r = currentPixel
    #Multiply respective numbers from kernel and current kernel(origin) in image
    upLeft = kernel[0][0] * fixedOriginal[c-1][r-1]#Upleft
    up = kernel[0][1] * fixedOriginal[c-1][r]#up
    upRight = kernel[0][2] * fixedOriginal[c-1][r+1]#upRight
    left = kernel[1][0] * fixedOriginal[c][r-1]#left
    origin = kernel[1][1] * fixedOriginal[c][r]#origin
    right = kernel[1][2] * fixedOriginal[c][r+1]#right
    downLeft = kernel[2][0] * fixedOriginal[c+1][r-1]#downLeft
    down = kernel[2][1] * fixedOriginal[c+1][r]#down
    downRight = kernel[2][2] * fixedOriginal[c+1][r+1]#downRight

    #add numbers together
    convolutedNumber = upLeft + upRight + up + left + origin + right + downLeft + down + downRight
    convolutedNumber = convolutedNumber / convolutionModifier
    return convolutedNumber

def reshapeToKernelSize(original,index):
    originalLength, originalWidth, other = np.shape(original)  # gets length of matrix
    removeLength = originalLength % len(kernel)  # shave this much from length
    removeWidth = originalWidth % len(kernel[0])  # shave this much from width

    fixedOriginal = np.empty(((originalLength - removeLength), originalWidth - removeWidth))
    fixedOriginal.fill(0)
    for x in range(originalLength - removeLength):
        for y in range(originalWidth - removeWidth):
            fixedOriginal[x][y] = original[x][y][index]

    return fixedOriginal

def extractRGBMatrices(): #returns the matrics for the red, blue, and green images of the original
    red = cpy(picture)
    green = cpy(picture)
    blue = cpy(picture)

    red[:,:,1] = 0 #sets all green in every pixel to 0
    red[:,:,2] = 0 #sets all blue in every pixel to 0

    green[:,:,0] = 0
    green[:,:,2] = 0

    blue[:,:,0] = 0
    blue[:,:,1] = 0

    return red, green, blue

if __name__=='__main__':
    picture = img.imread("albert.png")#Import image
    red, green, blue = extractRGBMatrices()

    redMat = np.asarray(red)
    greenMat = np.asarray(green)
    blueMat = np.asarray(blue)

    #grayImg = Image.open("noisyboi.png").convert("L")
    #original = np.asarray(grayImg) #intensity matrix
    #plt.imshow(blueMat, vmin= 0, vmax=255)
    plt.show()
    main()