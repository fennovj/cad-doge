# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Bram
"""

import skimage.io as skio
import cv2
from skimage import segmentation, color

import numpy as np

"""
Detect optic disc and makes it white, so you don't get confused with the
black background, needs the original rgb image
"""   
def detectOpticDisc(image):
    labels = segmentation.slic(image, n_segments = 70)
    out = color.label2rgb(labels, image, kind='avg')
    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    minimum = np.max(gray)
    image[gray==minimum] = 255
    return image

if __name__ == "__main__":

    imagepath = "C:\\Users\\Bram Arends\\Documents\\reduced"
    #samplepath = "D:\\Documents\\Dropbox\\CAD\\sample\\"
    #imagepath = "D:\\Downloads\\trainingdata\\train"
    #outfolder = "D:\\Downloads\\trainingdata\\reduced"
    
    testpic = "C:\\Users\\Bram Arends\\Documents\\reduced\\13_left.jpeg"
    testoutput = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\Project\\Detectionexamples"
    
    image = skio.imread(testpic, False)
    op = detectOpticDisc(image)
    skio.imshow(op)