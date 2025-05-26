import copy
import math
import time
import numpy as np
from collections import Counter
import random

# data = X_train, features = Y_train, cur = the current point inside X_test

def kFoldSplit(data, k=5, shuffle=True):
   indices = list(range(len(data)))
   if shuffle:
      random.shuffle(indices)
    
   foldSize = len(data) / k
   folds = []
   for i in range(k):
    # Calculate the start and end indices for the current fold
    start_i = i*foldSize
    if i != k-1:
        # For all folds except the last one, take exactly fold_size elements
        end_i = (i+1) * foldSize
    else:
        # For the last fold, include any remaining elements (in case len(data) is not divisible by k)
        end_i = len(indices)
    
    # Select the test indices for the current fold
    test_idx = indices[start_i:end_i]
    
    # Select the training indices by excluding the test indices
    train_idx = []
    for idx in indices:
      if idx not in test_idx:
         train_idx.append(idx)
    
    # Store the (train, test) indices as a tuple
    folds.append((train_idx, test_idx))
   return folds

def KFoldValidator(data, feature, current, alg, k=5):
    accuracy = 0
    test = copy.deepcopy(current)
    
    if alg == 0:  # forward selection
        test.append(feature)
    elif alg == 1 and feature != -1: # backward elimination
        test.remove(feature)
    elif alg == 1 and feature == -1: # no feature to remove for evaluation used for the beginning of backward
        for x in current:
            if x != feature:
                test.append(x)

    folds = kFoldSplit(data, k)

    total = 0
    for train_idx, test_idx in folds:
        for i in train_idx:
           train_data = data[i]

        correct = 0
        for i in test_idx:
            pred = NearestNeighbor(train_data + [data[i]], test, len(train_data))  # curr index is last in list
            if pred == data[i][0]:
                correct += 1

        accuracy += correct
        total += len(test_idx)

    return accuracy / total


def euclidean(point1, point2, features):
   distance = 0
   for feature in features:
      distance += (point1[feature] - point2[feature]) ** 2
   return np.sqrt(distance)

def NearestNeighbor(data, features, curr):
    neighborDist = math.inf
    nearest = 0
    for i in range(len(data)):
        if i != curr:
            distance = euclidean(data[curr], data[i], features)
            if distance <= neighborDist:
               # Set nearest neighbor to current data point
               neighborDist = distance
               # it stores the class label of the nearest data point
               nearest = data[i][0] 
    return nearest                            

def forwardSelection(data, features):
   return 0

def backwardElimination(data, features):
   return 0


"""
https://www.geeksforgeeks.org/k-nearest-neighbours/
https://www.geeksforgeeks.org/cross-validation-machine-learning/
https://www.kaggle.com/code/burhanykiyakoglu/k-nn-logistic-regression-k-fold-cv-from-scratch#-k-Nearest-Neighbors-(k-NN)
"""