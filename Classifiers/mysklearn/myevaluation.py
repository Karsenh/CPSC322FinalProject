"""
Luke Mason & Karsen Hansen
CPSC 322: Final Project
Apr 21 2021
myevaluation.py
"""

import mysklearn.myutils as myutils
import random
import math
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    # Check if the state is random to seed
    if random_state is not None:
       random.seed(random_state)
    
    # Check if the user wants to shuffle
    if shuffle: 
        for i in range(len(X)):
            rand_index = random.randrange(0, len(X))
            # Swamp indices the same way to maintain parallel lists
            X[i], X[rand_index] = X[rand_index], X[i]
            y[i], y[rand_index] = y[rand_index], y[i]
    
    n = len(X)
    if isinstance(test_size,float):
        test_size = math.ceil(n * test_size)
    split = n - test_size
    
    return X[:split], X[split:], y[:split], y[split:] # TODO: fix this

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size (n_samples // n_splits) + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n = len(X) # Get the length of the samples
    sample_sizes = [] # Array to be filled with fold sizes
    large_folds = n % n_splits # The amount of samples in the larger folds if data isn't even distributed

    # Loop through the folds
    for i in range(n_splits):
        # Check if the index is a large fold or a small fold
        if i >= large_folds:
            sample_sizes.append(n // n_splits)
        else:
            sample_sizes.append((n // n_splits) + 1)

    # Initialize lists to return
    X_train_folds = []
    X_test_folds = []

    # Loop through the folds
    for i in range(n_splits):
        indices = [k for k in range(len(X))] # Get the indices in the data
        range_size = sample_sizes[i] # Get the number of samples in this fold
        start_index = sum(sample_sizes[l] for l in range(i)) # Get the start index of the fold
        test_fold = [j for j in range(start_index, start_index + range_size)] # Get the test_fold
        X_test_folds.append(test_fold) # Append the test fold

        # Delete the test fold indices from the indices array
        del indices[start_index:start_index + range_size]
        X_train_folds.append(indices) # Append the train fold

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Make a copy of the data to combine labels
    x_copy = copy.deepcopy(X)

    # Add labels to input data
    for i in range(len(x_copy)):
        x_copy[i].append(y[i])
    
    # Group by the labels
    _, subtables = myutils.group_by(x_copy, -1)

    # Build up a list that contains the groups concatenated
    ungrouped = []
    for subtable in subtables:
        for i in subtable:
            ungrouped.append(i)

    # Spread the ungrouped list upon the folds
    folds = []
    # Init folds lists
    for i in range(n_splits):
        folds.append([])
    # Divvy up data
    for i in range(len(ungrouped)):
        index = i % n_splits
        folds[index].append(ungrouped[i])

    # Init return lists
    test_return = []
    train_return = []

    # Build up test and train lists
    for i in range(n_splits):
        folds_copy = copy.deepcopy(folds)
        test_sample = folds_copy[i] # Test gets the list at fold index
        train_sample = []
        del folds_copy[i] # Remove the list that is used for testing

        # Copy over the indices to train over
        for row in folds_copy:
            for index in row:
                train_sample.append(index)

        # Append built up lists to return
        test_return.append(test_sample)
        train_return.append(train_sample)
        
    return train_return, test_return

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [] # Initialize the matrix to return 

    # Fill the matrix with 0's
    for i in range(len(labels)):
        matrix_init_row = []
        for j in range(len(labels)):
            matrix_init_row.append(0)
        matrix.append(matrix_init_row)
    
    # Loop through the labels
    for i in range(len(labels)):
        label = labels[i] # Get the label to search over

        # Loop through the actual values
        for j in range(len(y_true)):

            # Check for match on label to index into matrix
            if y_true[j] == label:
                # Get index of predicted value in matrix row
                index = labels.index(y_pred[j])
                matrix[i][index] = matrix[i][index] + 1 # Increment count
    return matrix