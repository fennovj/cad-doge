# -*- coding: utf-8 -*-
"""
Created on Mon May 25 13:14:14 2015

@author: Fenno
"""

import numpy as np


class Queue:
    def __init__(self):
        self.items = []
        self.all = []
 
    def isEmpty(self):
        return self.items==[]
 
    def enque(self,item):
        self.items.insert(0,item)
        self.all.insert(0,item)
 
    def deque(self):
        return self.items.pop()
 
    def qsize(self):
        return len(self.items)
    
    def isInside(self, item):
        return (item in self.all)
        
fourneighbors = [lambda (x,y): (x+1, y),
                 lambda (x,y): (x-1, y),
                 lambda (x,y): (x, y+1),
                 lambda (x,y): (x, y-1)]


def regionGrowing(pic, seed, threshold, compare_original = True, neighbors = 4):
    """Given IxJ image and list of seed coordinates and list of thresholds,
    region grows all of them
    returns an image that is 1 at the grown regions, and 0 everywhere else
    """
    assert neighbors == 4, "Only 4 neighbors implemented right now"
    neighbors = fourneighbors #note: can add 8 neighbors later
    
    I, J = np.shape(pic)
    sx = seed[0]
    sy = seed[1]
    Q = Queue()
    Q.enque((sx,sy))
    
    while not Q.isEmpty():
        t = Q.deque()
        x = t[0]
        y = t[1]
        for neigh in neighbors:
            nx, ny = neigh((x,y))
            if not compare_original:  sx, sy = x, y #compare to current pixel or original pixel
            if (0 <= nx < I) and (0 <= ny < J) and \
               abs(pic[nx, ny] - pic[sx, sy]) <= threshold and \
               not Q.isInside((nx, ny)):
                Q.enque((nx, ny))    
    
    result = np.zeros((I,J))
    for (i, j) in Q.all:
        result[i,j] = 1
    return result
    
    
if __name__ == "__main__":
    fname = "D:\\Documents\\Data\\Cad Project\\reduced\\100_left.jpeg"
    import time
    import skimage.io as skio
    from skimage.viewer import ImageViewer
 
    #path = "D:\\Documents\\Data\\Cad Project\\reduced\\10003_left.jpeg"
    path = "C:\\Users\\Bram Arends\\Documents\\reduced\\16_left.jpeg"

    #path = "D:\\Documents\\Data\\Cad Project\\reduced\\rgb.bmp"
    image = skio.imread(path)
    
    result1 = regionGrowing(image[:,:,1], (0,0), 10, False)
    result2 = regionGrowing(image[:,:,1], (0,511), 10, False)
    result3 = regionGrowing(image[:,:,1], (511,0), 10, False)
    result4 = regionGrowing(image[:,:,1], (511,511), 10, False)
    result5 = np.logical_or(np.logical_or(result1, result2), np.logical_or(result3, result4))

    ImageViewer(result5).show()
               
            