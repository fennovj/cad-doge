# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:09:32 2015

@author: Fenno
"""

#The input for these features is typically a 512x512x3 matrix

import numpy as np
import sklearn.ensemble as ske
import skimage.io as skio
from os.path import join

def mask(picture, operation):
    mask = np.repeat(np.sum(picture, 2) == 0, 3, 2)
    return np.ma.masked_where(mask, picture)

def maskedIntensity(picture):
    intensity = np.mean(picture, 2)
    return np.ma.masked_where(intensity == 0, intensity)

mean_intensity = lambda intensity: np.ma.mean(intensity)
median_intensity = lambda intensity: np.ma.median(intensity)
std_intensity = lambda intensity: np.ma.std(intensity)
maskIntenFeatures = [mean_intensity, median_intensity, std_intensity]

def getFeatures(imagepath):
    image = skio.imread(imagepath)
    inten = maskedIntensity(image)
    return np.array([f(inten) for f in maskIntenFeatures])

def createFeatureMatrix(trainLabelsPath, imagespath, number = 10000, suffix = '.jpeg'):
    labels = np.genfromtxt(trainLabelsPath, dtype=None, delimiter = ',', skip_header = 1)
    assert 0 < number <= np.size(labels)
    labels = labels[:number]
    feats = getFeatures(join(imagespath, labels[0][0] + suffix))
    numfeats = np.size(feats)
    featMatrix = np.empty((number, numfeats))
    labs = np.empty(number)
    for i, (fname, label) in enumerate(labels):
        featMatrix[i,:] = getFeatures(join(imagespath, fname + suffix))
        labs[i] = label
    return featMatrix, labs.astype('int32')    
    
def randomForest(traindata, trainlabels, testdata, n_estimators = 50):
    model = ske.RandomForestClassifier(n_estimators = n_estimators).fit(traindata, trainlabels)
    return model.predict(testdata).astype('int32')
    
if __name__ == '__main__':
    #trainlabelpath = "D:\\Documents\\Dropbox\\CAD\\sampledata\\trainLabels.csv"
    trainlabelpath = "C:\\Users\\fenno_000\\Dropbox\CAD\\sampledata\\trainLabels.csv"
    imagepath = "C:\\Users\\fenno_000\\Documents\CAD Project\\reduced"
    feats, labels = createFeatureMatrix(trainlabelpath, imagepath, 100)
    print "done reading matrix"
    from crossvalidate import SampleCrossValidator
    validator = SampleCrossValidator(feats, labels, 0.1, 1.0)
    for train, classes, test in validator.yield_cross_validation_sets(rounds = 1):
        prediction = randomForest(train, classes, test, 50)
        validator.add_prediction(prediction)
    validator.print_results()
    
    