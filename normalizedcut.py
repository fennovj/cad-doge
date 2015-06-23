# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 13:14:14 2015

@author: Bram
"""

import numpy as np
from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt

def segmentBackground(image):
    labels = segmentation.slic(image, n_segments = 3)
    out = color.label2rgb(labels, image, kind='avg')
    intensity = np.mean(out, axis=2)
    minimum_intensity = np.min(intensity)
    image[intensity==minimum_intensity] = 0
    return image
    
if __name__ == "__main__":

    path = "C:\\Users\\Bram Arends\\Documents\\reduced\\280_left.jpeg"
    img = io.imread(path)
    
    #labels1 = segmentation.slic(img, compactness=30, n_segments=3)
    #labels1 = segmentation.slic(img, n_segments = 3)
    #out1 = color.label2rgb(labels1, img, kind='avg')
    
    #g = graph.rag_mean_color(img, labels1, mode='similarity')
    #labels2 = graph.cut_normalized(labels1, g)
    #out2 = color.label2rgb(labels2, img, kind='avg')
    out = segmentBackground(img)
    plt.figure()
    io.imshow(out)
    plt.figure()
    #io.imshow(out2)
    #io.show()