# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Bram
"""

import math
import skimage.io as skio
import cv2
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk, octagon, reconstruction, watershed, square
from skimage.filters import rank, threshold_otsu
from skimage import exposure, segmentation, color
from matplotlib import pyplot as plt


from scipy import ndimage

import numpy as np
import colorsys as color

def threshold(image, threshold = 0.5):
    image[image < threshold] = 0
    return image
    
def convertToHLS(image):
    rows, cols, channel = image.shape
    hlsimage = np.zeros((rows, cols, channel), dtype = float)
    image = image.astype(np.float)
    image = image/255.0
    for i in range(rows):
        for j in range(cols):
            hlsimage[i,j,:] = np.array(color.rgb_to_hls(*image[i,j,:]), dtype=float)
    return hlsimage
    
def convertToRGB(image):
    rows, cols, channel = image.shape
    rgbimage = np.zeros((rows, cols, channel), dtype = float)
    for i in range(rows):
        for j in range(cols):
            rgbimage[i,j,:] = np.array(color.hls_to_rgb(*image[i,j,:]), dtype=float)
    rgbimage = rgbimage*255.0
    return rgbimage.astype(np.uint8)   

def drawCircle(rows, cols, x, y, radius):
    im = np.zeros((rows, cols), dtype = np.int32)
    c = 4
    R = radius**2
    xp = 0
    yp = radius
    while (xp<yp):
        x1 = x+xp
        x2 = x-xp
        x3 = y+xp
        x4 = y-xp
        y1 = y+yp
        y2 = y-yp
        y3 = x+yp
        y4 = x-yp
        
        im[x1-c:x1+c, y2-c:y2+c] = 1
        im[x1-c:x1+c, y1-c:y1+c] = 1
        im[x2-c:x2+c, y2-c:y2+c] = 1
        im[x2-c:x2+c, y1-c:y1+c] = 1
        im[y3-c:y3+c, x4-c:x4+c] = 1
        im[y3-c:y3+c, x3-c:x3+c] = 1
        im[y4-c:y4+c, x4-c:x4+c] = 1
        im[y4-c:y4+c, x3-c:x3+c] = 1
        xp = xp + 1
        yp = math.sqrt(R-xp**2)
    im[x,y]=1
    return im
    
def computeCentroid(image):
    rows, cols = image.shape
    nrow = 0
    ncol = 0
    counter = 0
    for i in range(rows):
        for j in range(cols):
            if(image[i,j]>0):
                nrow = nrow + i
                ncol = ncol + j
                counter = counter + 1
    return nrow/counter, ncol/counter
    
def detectOpticDisc(image):
    kernel = octagon(10, 10)
    thresh = threshold_otsu(image[:,:,1])
    binary = image > thresh
    print binary.dtype
    luminance = convertToHLS(image)[:,:,2]
    t = threshold_otsu(luminance)
    t = erosion(luminance, kernel)
    
    
    labels = segmentation.slic(image[:,:,1], n_segments = 3)
    out = color.label2rgb(labels, image[:,:,1], kind='avg')
    skio.imshow(out)
    
    x, y = computeCentroid(t)
    print x, y
    rows, cols, _ = image.shape
    p1 = closing(image[:,:,1],kernel)
    p2 = opening(p1, kernel)
    p3 = reconstruction(p2, p1, 'dilation')
    p3 = p3.astype(np.uint8)
    #g = dilation(p3, kernel)-erosion(p3, kernel)
    #g = rank.gradient(p3, disk(5))
    g = cv2.morphologyEx(p3, cv2.MORPH_GRADIENT, kernel)
    #markers = rank.gradient(p3, disk(5)) < 10
    markers = drawCircle(rows, cols, x, y, 85)
    #markers = ndimage.label(markers)[0]
    #skio.imshow(markers)
    g = g.astype(np.uint8)
    #g = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    w = watershed(g, markers)
    print np.max(w), np.min(w)
    w = w.astype(np.uint8)
    #skio.imshow(w)
    return w
    
    
def detectVessels(image):
    kernel = square(2)
    image = ndimage.gaussian_filter(image[:,:,2], 2)
    image = opening(image, kernel)
    return image

if __name__ == "__main__":

    imagepath = "C:\\Users\\Bram Arends\\Documents\\reduced"
    #samplepath = "D:\\Documents\\Dropbox\\CAD\\sample\\"
    #imagepath = "D:\\Downloads\\trainingdata\\train"
    #outfolder = "D:\\Downloads\\trainingdata\\reduced"
    
    testpic = "C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"
    testoutput = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\Project\\Detectionexamples"
    
    image = skio.imread(testpic, False)
    #t = threshold(luminance, 0.75)    
    op = detectOpticDisc(image)
    print np.max(op), np.min(op)
    #skio.imshow(op)
    #skio.imshow(drawCircle(512, 512, 250, 150, 30))
    #skio.imshow(t)