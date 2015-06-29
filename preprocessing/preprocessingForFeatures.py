# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:14:26 2015

@author: Bram
"""

import skimage.io as skio
import numpy as np
import cv2

#from opticDiscVesselDetection import detectOpticDisc
from scipy.ndimage.filters import median_filter

"""
This class preprocesses the original image for feature extraction
"""
class Images(object):
    
    def __init__(self, image):
        self.green_plane = image[:,:,1]
        self.background_image = median_filter(self.green_plane, size=17)
        self.shade_corrected_image = self.green_plane.astype(np.int) - self.background_image.astype(np.int)
        self.pp = np.copy(self.shade_corrected_image)
        self.pp[self.pp > 0] = 0
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        intensity = np.mean(image, axis=2)
        intensity = median_filter(intensity, size=5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        intensity = intensity.astype(np.uint16)
        intensity = clahe.apply(intensity)
        intensity = intensity.astype(np.float)/float(255)
        r,c = intensity.shape
        self.hsi_image = np.concatenate((image_hsv[:,:,0:2], intensity.reshape(r,c,1)), axis=2)

if __name__ == "__main__":
    imagepath = "C:\\Users\\Bram Arends\\Documents\\reduced"    
    testpic = "C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"
    testoutput = "C:\\Users\\Bram Arends\\Dropbox\\CAD\\Project\\Detectionexamples"
    
    image = skio.imread(testpic, False)
    x = Images(image)
    skio.imshow(x.hsi_image[:,:,2])