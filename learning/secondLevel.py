__author__ = 'Fenno'

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np


#basic features: sum, mean, min, max, std

#connected components: max, biggest one

def bins(image):
    return np.bincount(image.flatten())[1:]


features = [np.max, np.mean, np.min, np.std, np.median ]

#connected components: ammount, biggest one, average size, average for substantial (> 10 pixels) components
componentfeatures = [np.max,
                       lambda x : np.max(bins(x)),
                       lambda x: np.mean(bins(x)),
                       lambda x: np.mean(bins(x)[bins(x) > 10])]

#Input: a 512x512xn matrix of predictions for either red or white lesion
#output: an nxf matrix of features
def getFeatures(imagematrix):
    _,_, n = np.shape(imagematrix)
    f = np.shape(features)
    result = np.zeros((n,f))
    for i in range(n):
        result[n,:] = [feature(imagematrix[:,:,n]) for feature in features]
    return result

def randomForestSecond(train,
                 labels,
                 test,
                 prior_weight = None,
                 n_estimators=100,
                 n_jobs=1,
                 verbose=0):
    """

    :param train: The features of training data, obtained with getFeatures
    :param labels: The kaggle labels of the training data
    :param test: The faetures of testing data
    :param prior_weight: the normalized weights to which output will be rescaled
            by default: no rescaling. If 'auto', use ratio from kaggle training data
    :param n_estimators:
    :param n_jobs:
    :param verbose:
    :return:
    """
    if prior_weight == 'auto':
        prior_weight = [25810/35126.0,  2443/35126.0,  5292/35126.0,   873/35126.0,   708/35126.0]
    assert np.sum(prior_weight) < (1.0 + 1e-4) and np.sum(prior_weight) > (1.0 - 1e-4)

    model = RandomForestRegressor(n_estimators=n_estimators,
                                   n_jobs=n_jobs,
                                   verbose=verbose)

    print "Now training model..."

    model.fit(train, labels)


    print "Now predicting samples..."
    predictions = model.predict_proba(test)

    if prior_weight is not None:
        sortedpred = np.sort(predictions)
        indexratio = np.cumsum(prior_weight)
        n = len(sortedpred)
        indexes = [int(i * n) for i in indexratio[:-1]]
        thresholds = [sortedpred[i] for i in indexes] + [sortedpred[-1]]
        predictions = np.digitize(predictions, thresholds, right=True)

    return predictions