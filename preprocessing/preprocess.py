# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Fenno
"""

import skimage.io as skio
from skimage.transform import resize
import numpy as np
import os
from skimage import segmentation, color

"""
Thresholds an image given a threshold, by setting all pixels below the threshold to 0
you are below the threshold if the mean of the RGP values < threshold
warning: depending on the threshold, will zero out pixels inside the eye as well 
TODO: fix this
"""
def threshold(image, threshold = 20.0):
    intensity = np.mean(image, axis=2)
    image[intensity < threshold] = 0
    return image

"""
Better alternative to thresholding
"""
def segmentBackground(image):
    labels = segmentation.slic(image, n_segments = 3)
    out = color.label2rgb(labels, image, kind='avg')
    intensity = np.mean(out, axis=2)
    minimum_intensity = np.min(intensity)
    image[intensity==minimum_intensity] = 0
    return image
    
"""
Makes an image square by adding empty rows to the top, bottom, left, right where needed,
then resizes the image to the desired size (by default, 256x256)
"""
def makeSquare(image, size = 256):
    rows, cols, _ = image.shape
    size = (size, size, 3)
    dtype = image.dtype
    if rows < cols:
        newrows = (cols-rows) / 2
        addstuff = np.zeros((newrows, cols, 3),dtype=dtype)
        image = np.concatenate((addstuff, image, addstuff), axis= 0)
        if cols - rows % 2:
            image = np.concatenate((image, np.zeros((1,cols,3),dtype=dtype)), axis=0)
    if cols < rows:
        newcols = (rows-cols) / 2
        addstuff = np.zeros((rows, newcols, 3),dtype=dtype)
        image = np.concatenate((addstuff, image, addstuff), axis=1)
        if rows - cols % 2:
            image = np.concatenate((image, np.zeros((rows,1,3),dtype=dtype)), axis=1)
    return resize(image, size)

"""
Crops an image by cutting off the rows and columns to the left, right, top, bottom
that are all zero. Therefore, it is smart to do this after thresholding
"""
def crop(image):
    nonzeros = np.nonzero(image)
    if np.size(nonzeros[0]) == 0 or np.size(nonzeros[0]) == 0:        
        return np.zeros((1,1,3),dtype=image.dtype)
    minrow = np.min(nonzeros[0])
    maxrow = np.max(nonzeros[0])
    mincol = np.min(nonzeros[1])
    maxcol = np.max(nonzeros[1])
    if (maxrow - minrow) == 0 or (maxcol - mincol) == 0:
        return np.zeros((1,1,3),dtype=image.dtype)
    return image[minrow:maxrow,mincol:maxcol,:]
    
def process(imagepath, outpath, size=512, thresh = 20.0):
    print "Now processing " + os.path.basename(imagepath)
    image = skio.imread(imagepath)
    image = segmentBackground(image)
    image = crop(image)
    image = makeSquare(image, size)
    skio.imsave(outpath, image)
    del image
    
if __name__ == "__main__":

    #imagepath = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\sample\\"
    samplepath = "D:\\Documents\\Dropbox\\CAD\\sample\\"
    imagepath = "D:\\Downloads\\trainingdata\\train"
    outfolder = "D:\\Downloads\\trainingdata\\reduced"
    
    testpic = "D:\\Downloads\\trainingdata\\reduced\\11023_left.jpeg"
    testoutput = "D:\\Downloads\\trainingdata\\test.jpeg"
    #process(testpic, testoutput, 512, 20.0)
    
    #process(samplepath + "16_left.jpeg", samplepath + "test.jpeg", 512, 20.0 )

    images = os.listdir(imagepath)
    done = os.listdir(outfolder)
    for picture in images:
        if picture not in done:
            process(os.path.join(imagepath, picture), os.path.join(outfolder, picture))

    #total: 35126 images