#!/usr/bin/env python3
import cv2 as cv
import numpy as np



def main():
    sharp1 = np.zeros_like(img)

    for i in range(3): #For each color
        sharp1[:, :, i] = laplacian(img[:, :, i], .8)

    while cv.waitKey(1) < 0:
        cv.imshow('Original', img)
        cv.imshow('Laplace', sharp1)


def laplacian(image, strength):
    gaussian = cv.GaussianBlur(image, (3, 3), 0) #Apply noise filter

    laplacian = cv.Laplacian(gaussian, cv.CV_64F) #Apply edge detector

    sharp = image - (strength * laplacian) #Sharpen the original image by subtracting the negative edges
    sharp[sharp > 255] = 255 #Set cap of pixels to 255
    sharp[sharp < 0] = 0 #Set lower-limit cap to 0

    return sharp

if __name__ == '__main__':
    img = cv.imread('../images/lena.png')


    main()