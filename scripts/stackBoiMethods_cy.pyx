import cv2 as cv
import torch
import numpy as np

"""
@description Stacks the images values into blue green and red images
@returns Arrays for blues, greens, reds containing all the value from each picture
"""
def stackImages(images, cuda):
    cdef int x = 0
    cdef int h = 0
    cdef int w = 0
    cdef int imageAmount = images.shape[0]
    cdef int resizedHeight = images.shape[1]
    cdef int resizedWidth = images.shape[2]

    images = images.cpu().numpy()
    #Make arrays to store stacked images for RGB
    blues = np.empty(((resizedHeight, resizedWidth, imageAmount)), dtype = np.uint8)
    greens = np.empty(((resizedHeight, resizedWidth, imageAmount)), dtype = np.uint8)
    reds = np.empty(((resizedHeight, resizedWidth, imageAmount)), dtype = np.uint8)

    print(images.dtype)
    print("Images: ", images.shape[0])
   
    for x in range(imageAmount): #loop through all the images
        print("Stacking image %i" %(x))
        for h in range(resizedHeight): #Loops through the all pixels in the images
            for w in range(resizedWidth):
                blues[h][w][x] = images[x][h][w][0]
                greens[h][w][x] = images[x][h][w][1]
                reds[h][w][x] = images[x][h][w][2]

    blues = torch.from_numpy(blues)
    greens = torch.from_numpy(greens)
    reds = torch.from_numpy(reds)
    return blues, greens, reds


"""
@Description Finds the median from each of the BGR Images
@returns the median image from all the images
"""
def medianize(blues, greens, reds, images):
    cdef int avgBlue = 0
    cdef int avgGreen = 0
    cdef int avgRed = 0
    cdef int h = 0
    cdef int w = 0
    cdef int newLength = blues.shape[0]
    cdef int newWidth = blues.shape[1]
    cdef int colorAmount = 3

    lastImage = np.empty(((newLength, newWidth, colorAmount)), np.uint8)

    


    for h in range(newLength): #Loops through the all pixels in the images
        print("Row: ", h)
        for w in range(newWidth): #Gets the median count for all the images
            avgBlue = torch.median(blues[h][w])
            avgGreen = torch.median(greens[h][w])
            avgRed = torch.median(reds[h][w])

            lastImage[h][w][0] = avgBlue
            lastImage[h][w][1] = avgGreen
            lastImage[h][w][2] = avgRed     

    return lastImage