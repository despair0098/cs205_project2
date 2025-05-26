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

def forwardSelection(data, features, k=5):
    current_set = []
    best_set = []
    best_accuracy = 0.0

    print("Beginning forward selection with {}-fold cross-validation.".format(k))

    for level in range(1, len(features) + 1):
        start = time.time()
        print(f"\nOn level {level} of the search tree...")

        feature_to_add = None
        best_so_far_accuracy = 0.0

        for feature in features:
            if feature in current_set:
                continue  # Skip already selected features

            # Evaluate accuracy with the candidate feature added
            accuracy = KFoldValidator(data, feature, current_set, 0, k)
            candidate_set = current_set + [feature]
            print(f"Using feature(s) {candidate_set}, accuracy is {accuracy:.4f}")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_add = feature

        if feature_to_add is not None:
            current_set.append(feature_to_add)
            print(f"Feature {feature_to_add} added to the current set.")
        else:
            print("No feature improved accuracy at this level.")

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_set = current_set[:]
        else:
            print("Warning: Accuracy has decreased or plateaued.")

        print(f"Current best feature set: {current_set} with accuracy {best_so_far_accuracy:.4f}")
        print(f"Time to evaluate level {level}: {time.time() - start:.2f} seconds")

    print("\nFinished search!")
    print(f"The best feature subset is {best_set}, with an accuracy of {best_accuracy:.4f}")

def backwardElimination(data, features, k=5):
    current_set = features[:]  # Start with all features
    best_set = features[:]
    best_accuracy = KFoldValidator(data, -1, current_set, 1, k)

    print(f"Starting Backward Elimination with features: {current_set}")
    print(f"Initial full-set accuracy: {best_accuracy:.4f}\n")

    for level in range(1, len(features) + 1):
        start = time.time()
        print(f"Level {level}:")

        feature_to_remove = None
        best_so_far_accuracy = 0.0

        for feature in current_set:
            candidate_set = current_set[:]
            candidate_set.remove(feature)

            # Evaluate candidate set (pass original set and feature to remove)
            accuracy = KFoldValidator(data, feature, current_set, alg=1, k=k)

            print(f"  Using feature(s) {candidate_set}, accuracy is {accuracy:.4f}")

            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                feature_to_remove = feature

        if feature_to_remove is not None:
            current_set.remove(feature_to_remove)
            print(f"Removed feature {feature_to_remove}.")
        else:
            print("No feature removal improved accuracy.")

        if best_so_far_accuracy > best_accuracy:
            best_set = current_set[:]
            best_accuracy = best_so_far_accuracy
        else:
            print("Warning: Accuracy has decreased or plateaued.")

        print(f"Current best feature set: {current_set} with accuracy {best_so_far_accuracy:.4f}")
        print(f"Time to evaluate level {level}: {time.time() - start:.2f} seconds\n")

    print("Finished search!")
    print(f"The best feature subset is {best_set} with accuracy {best_accuracy:.4f}")


"""
https://www.geeksforgeeks.org/k-nearest-neighbours/
https://www.geeksforgeeks.org/cross-validation-machine-learning/
https://www.kaggle.com/code/burhanykiyakoglu/k-nn-logistic-regression-k-fold-cv-from-scratch#-k-Nearest-Neighbors-(k-NN)
#https://stackoverflow.com/questions/35608790/how-to-add-new-value-to-a-list-without-using-append-and-then-store-the-value
#https://stackoverflow.com/questions/11993878/python-why-does-my-list-change-when-im-not-actually-changing-it
"""