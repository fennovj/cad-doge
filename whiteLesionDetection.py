# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Bram
"""

import math
import skimage.io as skio
import numpy as np
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, octagon, reconstruction, watershed, square
from opticDiscVesselDetection import opticDiscDetection
from scipy.ndimage.filters import median_filter
from sklearn.cluster import AgglomerativeClustering



def featureExtraction(hsv_pixel, rgb_pixel, neighborhood, ):
    return [hsv_pixel[2], hsv_pixel[0]]
    

def detectWhiteLesions(image):
    op = opticDiscDetection(image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    f1 = median_filter(img_hsv[:,:,2], 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    f2 = clahe.apply(f1)
    image_hsv[:,:,2] = f2
    
    return image

if __name__ == "__main__":

    imagepath = "C:\\Users\\Bram Arends\\Documents\\reduced"
    #samplepath = "D:\\Documents\\Dropbox\\CAD\\sample\\"
    #imagepath = "D:\\Downloads\\trainingdata\\train"
    #outfolder = "D:\\Downloads\\trainingdata\\reduced"
    
    testpic = "C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"
    testoutput = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\Project\\Detectionexamples"
    
    image = skio.imread(testpic, False)
    #img_hsv = convertToHLS(image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    f1 = median_filter(img_hsv[:,:,2], 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #clahe = cv2.createCLAHE()
    f2 = clahe.apply(f1)
    #f2 = cv2.equalizeHist(f1)
    #t = threshold(luminance, 0.75)    
    #op = detectOpticDisc(image)
    skio.imshow(f2)
    #skio.imshow(drawCircle(512, 512, 250, 150, 30))
    #skio.imshow(t)