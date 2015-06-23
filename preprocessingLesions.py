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
from opticDiscVesselDetection import detectOpticDisc
from scipy.ndimage.filters import median_filter
from sklearn.cluster import AgglomerativeClustering

class Images(object):
    
    def __init__(self, image):
        self.green_plane = image[:,:,1]
        self.background_image = median_filter(self.green_plane, size=17)
        self.shade_corrected_image = self.green_plane.astype(np.int) - self.background_image.astype(np.int)
        print self.shade_corrected_image.dtype
        self.pp = np.copy(self.shade_corrected_image)
        self.pp[self.pp > 0] = 0
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        intensity = np.sum(image, axis=2)/float(3)
        intensity = median_filter(intensity, size=25)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        intensity = intensity.astype(np.uint16)
        intensity = clahe.apply(intensity)
        intensity = intensity.astype(np.float)/float(255)
        r,c = intensity.shape
        self.hsi_image = np.concatenate((image_hsv[:,:,0:2], intensity.reshape(r,c,1)), axis=2)

if __name__ == "__main__":
    imagepath = "C:\\Users\\Bram Arends\\Documents\\reduced"    
    testpic = '16_left.jpeg'#"C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"
    testoutput = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\Project\\Detectionexamples"
    
    
    from skimage.viewer import ImageViewer
    image = skio.imread(testpic, False)
    x = Images(image)
    skio.imshow(x.background_image)
    skio.imshow(x.hsi_image)
    
    ImageViewer(x.green_plane).show()
    ImageViewer(x.background_image).show()
    ImageViewer(x.hsi_image).show()
    ImageViewer(x.pp).show()
