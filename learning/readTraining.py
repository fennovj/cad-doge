__author__ = 'Fenno'
import os
import skimage.io as skio
from sklearn.ensemble import RandomForestClassifier
import numpy as np

BASEFOLDER = 'D:\\Documents\\Data\\DMED'
IMGFOLDER = 'DMED-P'
LABELFOLDER = 'DMED-P'
NAMEFOLDER = 'DMED-GT'

TESTFOLDER = 'Reduced'

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
                 max_leaf_nodes=None,
                 n_jobs=1,
                 verbose=0):
    """
    Trains a model by giving it a feature matrix, as well as the labels (the ground truth)
    then using that model, predicts the given test samples
    output is 9 probabilities, one for each class
    :param train: The training data, to train the model
    :param labels: The labels of the training data, an array
    """
    n,f = np.shape(test)
    assert n % (512*512) == 0
    numimg = int(n / (512*512))

    model = RandomForestClassifier(n_estimators=n_estimators,
                                   criterion=criterion,
                                   max_features=max_features,
                                   max_depth=max_depth,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_leaf_nodes=max_leaf_nodes,
                                   n_jobs=n_jobs,
                                   verbose=verbose)

    model.fit(train, labels, sample_weight)
    predictions = model.predict_proba(test)[:,1]
    return predictions.reshape((512,512,numimg))

def readSingleTraining(name, path = os.path.join(BASEFOLDER, IMGFOLDER)):
    picture = skio.imread(os.path.join(path, name) + '.jpg')
    labels = skio.imread(os.path.join(path, name) + '.tif')
    return picture, labels

#Assumed pre-processed images are in DMED-P
#Assumes there are only .jpg and .tif files in that directory, and no other files
def readAllTraining(path = os.path.join(BASEFOLDER, IMGFOLDER)):
    numimage = len(os.listdir(path)) / 2
    result = np.zeros((512, 512, 3, numimage))
    resultlabels = np.zeros((512, 512, numimage))
    for i, filename in enumerate(os.listdir(path)):
        if filename.split('.')[-1] == 'tif':
            continue
        f = os.path.basename(filename).split('.')[0]
        result[:,:,:,i/2], resultlabels[:,:,i/2] = readSingleTraining(f, path)

    return result, resultlabels

def readSingleTesting(name, path):
    picture = skio.imread(os.path.join(path, name) + '.jpeg')
    return picture

def readAllTesting(path= os.path.join(BASEFOLDER, TESTFOLDER), num=100):
    numimage = len(os.listdir(path))
    result = np.zeros((512, 512, 3, num))
    for i, filename in enumerate(os.listdir(path)):
        if i == num:
            break
        f = os.path.basename(filename).split('.')[0]
        result[:,:,:,i] = readSingleTesting(f, path)
    return result

from whitelesionfeatures import getWhiteLesionFeatures as FEATUREEXTRACTOR

def getFeatureMatrix(imagematrix, labelmatrix = None, featureExtractor = FEATUREEXTRACTOR ):
    x,y,d,n = np.shape(imagematrix)
    assert x == 512 and y == 512 and d == 3
    print "n = ", n

    if labelmatrix is not None:
        x2, y2, n2 = np.shape(labelmatrix)
        assert x2 == 512 and y2 == 512 and n == n2
        labelmatrix = np.copy(labelmatrix)
        labelmatrix[labelmatrix > 0] = 1
    for i in range(n):
        print "Now at image " + str(i)
        features = featureExtractor(imagematrix[:,:,:,i])
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


if __name__ == '__main__':
    #imagematrix, labelmatrix = readAllTraining()
    #imagematrix = readAllTesting()
    #imagematrix = imagematrix.astype('uint8')
    #resultfeatures = getFeatureMatrix(imagematrix)
    #np.save("resultfeaturestest", resultfeatures)
    #np.save("resultlabels", resultlabels)

    train = np.load('resultfeatures.npy')
    labels = np.load('resultlabels.npy')
    test = np.load('resultfeaturestest.npy')
    predictions = randomForest(train, labels, test, n_estimators=200, n_jobs=2, verbose=1)
    np.save("predictions", predictions)