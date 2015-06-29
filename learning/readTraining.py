__author__ = 'Fenno'
import os
import skimage.io as skio
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BASEFOLDER = 'D:\\Documents\\Data\\DMED'
IMGFOLDER = 'DMED-P'
LABELFOLDER = 'DMED-P'
TXTFOLDER = 'DMED-GT'

#assumes preprocessed images are in DMED-P
def getTrainingImage(name, words = ['Exudate'], basefolder = BASEFOLDER, imgfolder = IMGFOLDER, labelfolder = LABELFOLDER, namefolder = NAMEFOLDER):
    imagepath = os.path.join(basefolder, imgfolder, name + '.jpg')
    labelspath = os.path.join(basefolder, labelfolder, name+ '.tif')
    namespath = os.path.join(basefolder, namefolder, name + '.txt')

    image = skio.imread(imagepath)
    labels = skio.imread(labelspath)

    with open(namespath) as f:
        for line in f:
            num = int(line.split(':')[0])
            if not any(x in line for x in words):
                labels[labels == num] = 0

    return image, labels


def randomForest(train,
                 labels,
                 test,
                 sample_weight=None,
                 n_estimators=100,
                 criterion='gini',
                 max_features="auto",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_leaf_nodes=None,
                 n_jobs=1,
                 verbose=0,
                 class_weight=None):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    :param train: The training data, to train the model
    :param labels: The labels of the training data, an array
    """
    n,f = np.shape(test)
    assert f % (512*512) == 0
    numimg = int(f / (512*512))

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   criterion=criterion,
                                   max_features=max_features,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   min_weight_fraction_leaf=min_weight_fraction_leaf,
                                   max_leaf_nodes=max_leaf_nodes,
                                   n_jobs=n_jobs,
                                   verbose=verbose,
                                   class_weight=class_weight)

    model.fit(train, labels, sample_weight)
    predictions = model.predict_proba(test)[:,1]
    return predictions.reshape((512,512,numimg))

#Assumed pre-processed images are in DMED-P
def readAllTraining(words = ['Exudate'], basefolder = BASEFOLDER):
    path = os.path.join(basefolder, 'DMED-P')
    numimage = len(os.listdir(path))
    result = np.zeros((512, 512, numimage))
    resultlabels = np.zeros((512, 512, numimage))
    for i, filename in enumerate(os.listdir(path)):
        f = os.path.splitext(filename)[0]
        result[:,:,i], resultlabels[:,:,i] = getTrainingImage(f, words, basefolder)

    return result, resultlabels

from whitelesionfeatures import getWhiteLesionFeatures as FEATUREEXTRACTOR

def getFeatureMatrix(imagematrix, labelmatrix = None, featureExtractor = FEATUREEXTRACTOR ):
    x,y,n = np.shape(imagematrix)
    assert x == 512 and y == 512
    if labelmatrix is not None:
        labelmatrix = np.copy(labelmatrix)
        labelmatrix[labelmatrix > 0] = 1
    for i in range(n):
        features = featureExtractor(imagematrix[:,:,i])
        _, f = np.shape(features)
        if i == 0:
            resultfeatures = np.empty((0,f))
        resultfeatures = np.vstack((resultfeatures, features))
        if labelmatrix is not None:
            if i == 0:
                resultlabels = np.empty(0)
            resultlabels = np.concatenate((resultlabels, np.ravel(labelmatrix[:,:,i])))
    if labelmatrix is not None:
        return resultfeatures, resultlabels
    return resultfeatures

