# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Fenno
"""

import skimage.io as skio
from skimage.transform import resize
from scipy.ndimage.interpolation import zoom
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
def makeSquare(image, size=512, labels=None):
    rows, cols, _ = image.shape
    size = (size, size, 3)
    dtype = image.dtype
    if labels is not None:
        labeldtype = labels.dtype
    if rows < cols:
        newrows = (cols-rows) / 2
        addstuff = np.zeros((newrows, cols, 3),dtype=dtype)
        image = np.concatenate((addstuff, image, addstuff), axis= 0)
        if cols - rows % 2:
            image = np.concatenate((image, np.zeros((1,cols,3),dtype=dtype)), axis=0)

        if labels is not None:
            #print np.shape(labels), np.shape(addstuff), np.shape(addstuff[:,:,0])
            labels = np.concatenate((addstuff[:,:,0], labels, addstuff[:,:,0]), axis = 0)
            if cols - rows % 2:
                labels = np.concatenate((labels, np.zeros((1,cols), dtype=labeldtype)),axis=0)
    if cols < rows:
        newcols = (rows-cols) / 2
        addstuff = np.zeros((rows, newcols, 3),dtype=dtype)
        image = np.concatenate((addstuff, image, addstuff), axis=1)
        if rows - cols % 2:
            image = np.concatenate((image, np.zeros((rows,1,3),dtype=dtype)), axis=1)

        if labels is not None:
            #print np.shape(labels), np.shape(addstuff), np.shape(addstuff[:,:,0])
            labels = np.concatenate((addstuff[:,:,0], labels, addstuff[:,:,0]), axis = 1)
            if cols - rows % 2:
                labels = np.concatenate((labels, np.zeros((rows,1), dtype=labeldtype)),axis=1)

    #skio.imsave(os.path.join("D:\\Documents\\Data\\DMED", "DMED-P", "test5.tif"), labels)
    if labels is not None:
        return resize(image, size), zoom(labels, 512.0 / max(rows, cols))
    return resize(image, size)

"""
Crops an image by cutting off the rows and columns to the left, right, top, bottom
that are all zero. Therefore, it is smart to do this after thresholding
"""
def crop(image, labels = None):
    nonzeros = np.nonzero(image)
    if np.size(nonzeros[0]) == 0 or np.size(nonzeros[0]) == 0:        
        return np.zeros((1,1,3),dtype=image.dtype)
    minrow = np.min(nonzeros[0])
    maxrow = np.max(nonzeros[0])
    mincol = np.min(nonzeros[1])
    maxcol = np.max(nonzeros[1])
    if (maxrow - minrow) == 0 or (maxcol - mincol) == 0:
        return np.zeros((1,1,3),dtype=image.dtype)
    if labels is not None:
        return image[minrow:maxrow,mincol:maxcol,:], labels[minrow:maxrow,mincol:maxcol]
    return image[minrow:maxrow,mincol:maxcol,:]
    
def process(imagepath, outpath, size=512):
    print "Now processing " + os.path.basename(imagepath)
    image = skio.imread(imagepath)
    image = segmentBackground(image)
    image = crop(image)
    image = makeSquare(image, size)
    skio.imsave(outpath, image)
    del image

from learning.readTraining import getTrainingImage

def processLabels(name, outpath, labelout, size=512):
    #image = skio.imread(imagepath)
    #labels = skio.imread(labelpath)
    image, labels = getTrainingImage(name)
    labels[labels > 0] = -1
    image = segmentBackground(image)
    image, labels = crop(image, labels)
    image, labels = makeSquare(image, size, labels)
    #skio.imsave(os.path.join("D:\\Documents\\Data\\DMED", "DMED-P", "test6.tif"), labels)
    skio.imsave(outpath, image)
    skio.imsave(labelout, labels)
    
if __name__ == "__main__":

    """

    BASEFOLDER = "D:\Documents\Data\CAD Project\\trainingdata"
    imagepath = os.path.join(BASEFOLDER, "train")
    outfolder = os.path.join(BASEFOLDER, "reducednew")

    images = os.listdir(imagepath)
    done = os.listdir(outfolder)
    for picture in images:
        if picture not in done and os.path.splitext(picture)[-1] in ['.jpg', '.jpeg']:
            process(os.path.join(imagepath, picture), os.path.join(outfolder, picture))

    """

    BASEFOLDER = "D:\\Documents\\Data\\DMED"
    imagepath = os.path.join(BASEFOLDER, "DMED")
    outfolder = os.path.join(BASEFOLDER, "DMED-P")
    labelpath = os.path.join(BASEFOLDER, "DMED-GT")
    images = os.listdir(imagepath)
    done = os.listdir(outfolder)
    for picture in images:
        if picture not in done and os.path.splitext(picture)[-1] in ['.jpg', '.jpeg']:
            label = ''.join(os.path.splitext(picture)[:-1]) + '.tif'
            processLabels(os.path.basename(picture).split('.')[0], os.path.join(outfolder, picture), os.path.join(outfolder, label))
            #processLabels(os.path.join(imagepath, picture), os.path.join(labelpath, label) , os.path.join(outfolder, picture), os.path.join(outfolder, label))