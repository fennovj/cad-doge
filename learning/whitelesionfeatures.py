# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:14:26 2015

@author: Bram
"""

import sys
import numpy as np
from skimage import io as skio, color
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from scipy.stats.mstats import zscore
from skimage.filters import sobel
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering
import cv2

sys.path.append("C:\Users\Bram Arends\Dropbox\CAD\Project\cad-doge\preprocessing")
from preprocessing.preprocessingForFeatures import Images
from preprocessing.opticdiscdetection import detectOpticDisc

"""
Areas of clusters (number of pixels)
"""
def getAreas(labels):
    labeling = np.unique(labels)
    for label in labeling:
        labels[labels==label*np.ones(labels.shape)] = np.sum(labels==label*np.ones(labels.shape))
    return labels

"""
Get features corresponding to agglomerative clustering
"""
def agglomerativeClusteringFeatures(image):
    connectivity = grid_to_graph(*image[:,:,2].shape)
    X = np.reshape(image[:,:,2], (-1,1))
    ward = AgglomerativeClustering(n_clusters=150,
        linkage = 'ward', connectivity = connectivity).fit(X)
    labels = np.reshape(ward.labels_, image[:,:,2].shape)
    averageIntensity = color.label2rgb(labels, image[:,:,2], kind = 'avg')
    #areas = getAreas(labels) 
    return averageIntensity

"""
Compute ratio between optic disc and area of clusters
"""
def computeRatio(op, areas):
      r = op[:,:,0]==255*np.ones(op[:,:,0].shape)  
      g = op[:,:,1]==255*np.ones(op[:,:,1].shape)  
      b = op[:,:,2]==255*np.ones(op[:,:,2].shape)
      boolean = np.logical_and(np.logical_and(r, g), b)
      sizeOp = np.sum(boolean)
      return areas/sizeOp
    
"""
Get features corresponding to the optic disc, needs the size of the 
clusters as well
"""
def opticDiscFeatures(rgbimage, areas):
    op = detectOpticDisc(rgbimage)
    ratios = computeRatio(op, areas)
    return ratios
    
"""
TODO
"""
def nrOfEdgePixels(rgbimage, intensityImage):
    redEdges = sobel(image[:,:,0])
    grayEdges = sobel(intensityImage)
    t = redEdges - grayEdges
    t[t < 0.05] = 0
    t[t >= 0.05] = 1
    return convolve2d(t, np.ones((17,17)), mode="same")

"""
The standard deviation of the preprocessed intensity
values in a window around the pixel
"""
def stdConvoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im**2
    ones = np.ones(im.shape)

    kernel = np.ones((2*N+1, 2*N+1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")

    return np.sqrt((s2 - s**2 / ns) / ns)

def getDoG(image, sigma1, sigma2):
    f1 = gaussian_filter(image, sigma1)
    f2 = gaussian_filter(image, sigma2)
    return f2-f1
    
"""
Get all the features for white lesion detection
"""
def getWhiteLesionFeatures(image):
    ppimages = Images(image)
    areas = agglomerativeClusteringFeatures(ppimages.hsi_image)
    features = np.concatenate((ppimages.hsi_image[:, :, 2].reshape((512*512, 1)), 
            stdConvoluted(ppimages.hsi_image[:,:,2], 7).reshape((512*512, 1)), 
            ppimages.hsi_image[:, :, 0].reshape((512*512, 1)), 
            opticDiscFeatures(image, areas).reshape((512*512, 1)), 
            nrOfEdgePixels(image, ppimages.hsi_image[:, :, 2]).reshape((512*512, 1)),
            getDoG(ppimages.hsi_image[:,:,2], 4, 8).reshape((512*512, 1))),
            axis=1)
    return zscore(features)
    
if __name__ == "__main__":

    testpic = "C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"
    image = skio.imread(testpic, False)
    features = getWhiteLesionFeatures(image)
    print np.min(features), np.max(features)
    print features.shape

    #skio.imshow(test.astype(np.uint8))
    #out = getWhiteLesionFeatures(image)
    