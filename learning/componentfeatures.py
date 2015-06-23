# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:46:22 2015

@author: Fenno
"""

import numpy as np
from skimage.measure import perimeter as perim
from math import pi

area = lambda picture, label: np.sum(picture == label)
    
perimeter = lambda picture, label: perim(picture == label, 8)
    
def aspectratio(picture, label):
    widtho = np.max(picture == label, 0)
    heighto = np.max(picture == label, 1)
    minw = np.argmax(widtho)
    minh = np.argmax(heighto)
    maxw = len(widtho) - np.argmax(widtho[::-1]) - 1
    maxh = len(heighto) - np.argmax(heighto[::-1]) - 1
    return (maxh - minh) / (1.0 * (maxw - minw))
    
circularity = lambda picture, label: perimeter(picture, label) / (4.0 * pi * area(picture, label))

totalgreen = lambda picture, label, green: np.sum(green[picture==label])
#can also be used for shadecorrected

meangreen = lambda picture, label, green: np.mean(green[picture==label])
#can also be used for shadecorrected image

normalizedintensity = lambda picture, label, green, bg : np.mean(green - np.mean(bg)) / np.std(bg)
#can also be used for shadecorrected

normmeanintensity = lambda picture, label, green, bg : (np.mean(green) - np.mean(bg)) / np.std(bg)
#can also be used for shadecorrected

#centroid and edgedist are not actual features
centroid = lambda picture, label : np.mean(np.where(picture==label),axis=1) 

def edgedist(pixel, shape=(512,512)):
    x = pixel[0]
    y= pixel[1]
    circumference = 2.0*shape[0] + 2.0*shape[1]
    totaldist = np.sum([np.hypot(x-a, y-b) for a in range(shape[0]) for b in [0,shape[1]]])
    totaldist2 = np.sum([np.hypot(x-a, y-b) for a in [0,shape[0]] for b in range(shape[1])])
    return (totaldist + totaldist2) / circumference, circumference

def compactness(picture, label):
    c = centroid(picture, label)
    db, circum = edgedist(c, np.shape(picture))
    pixels = np.array(np.where(picture==label)).T
    area = np.shape(pixels)[0]
    totaldist = np.sum([np.hypot(*(pixels[i,:]-c)) for i in range(area)])
    return np.sqrt((totaldist - db) / float(circum))

def getDiameter(pixel, shape=(512,512)):
    return np.min([pixel[0], pixel[1], shape[0] - pixel[0], shape[1]-pixel[1]])

