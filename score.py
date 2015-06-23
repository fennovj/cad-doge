
"""
	Functions for assessing the performance of different classifiers.
"""

import numpy as np

def pad_to_length(array, length, value = 0):
    addedzeros = length - np.size(array)
    return np.pad(array,(0,addedzeros), 'constant', constant_values = value)

def calc_weighted_kappa(predictions, true_classes):
    """
        The accuracy of the predictions (how many of the predictions were correct).
		inputs must be arrays/lists, consisting of integers (either signed or unsigned)
        inputs must be one-dimensional arrays, with the same length
        predictions must be integers.
    	   :return: The quadratic weighted kappa score
    """
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)
    if not isinstance(true_classes, np.ndarray):
        true_classes = np.array(true_classes)
    assert np.size(np.shape(predictions)) == 1 and np.size(np.shape(true_classes)) == 1
    assert np.size(predictions) == np.size(true_classes)
    assert predictions.dtype.kind in 'ui' and true_classes.dtype.kind in 'ui'
    N = np.amax(np.concatenate((predictions, true_classes))) + 1 #number of classes, starting from 0
    O = np.zeros((N,N),dtype='float')
    for i in range(np.size(predictions)):
        O[predictions[i], true_classes[i]] += 1.0
    W = np.array([[((i - j)**2.0)/((N - 1)**2.0) for j in range(0, N)]for i in range(0, N)])
    predbin = pad_to_length(np.bincount(predictions), N)
    truebin = pad_to_length(np.bincount(true_classes), N)
    E = np.outer(predbin, truebin)
    E = E * (np.sum(O) / np.sum(E))
    return 1.0 - (np.sum(W * O) / np.sum(W * E))

def calc_accuracy(predictions, true_classes):
	"""
		The accuracy of the predictions (how many of the predictions were correct).

		Predictions: the predicted classes
		true_classes: the true classes
		:return: The accuracy as a fraction [0-1].
	"""
	assert np.size(np.shape(predictions)) == 1 and np.size(np.shape(true_classes)) == 1
	assert np.size(predictions) == np.size(true_classes)
	N = float(np.size(predictions))
	return 1.0 - (np.count_nonzero(predictions-true_classes) / N)


if __name__ == '__main__':
    x = np.array([1,2,3,0])
    y = np.array([1,2,3,4])
    N = np.amax(np.concatenate((x,y))) + 1
    print "N = " + str(N)
    print calc_weighted_kappa( [1,1,2,3,2,1,0,0,1,2,3,4,4,3,2,1,0],[1,4,4,2,1,2,3,4,1,0,0,2,0,1,2,3,0])