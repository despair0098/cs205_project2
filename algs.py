import copy
import math
import time
import numpy as np
from collections import Counter

# data = X_train, features = Y_train, cur = the current point inside X_test

def KFold(data, k):
   

   return 0 

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

