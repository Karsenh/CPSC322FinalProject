import mysklearn.myutils as myutils
import math
import random
from collections import Counter


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
    X_train = X_test = y_train = y_test = None
    if random_state is not None:
        # TODO: seed your random number generator
        # you can use the math module or use numpy for your generator
        # choose one and consistently use that generator throughout your code
        random.seed(random_state)
    if shuffle:
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still
        # implement this and check your work yourself
        combined = [(x, y) for x, y in zip(X, y)]
        random.shuffle(combined)
        X = [i[0] for i in combined]
        y = [i[1] for i in combined]
    if isinstance(test_size, int):
        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]
    elif isinstance(test_size, float):
        length = math.ceil(len(X) * test_size)
        X_train = X[:-length]
        X_test = X[-length:]
        y_train = y[:-length]
        y_test = y[-length:]
    return X_train, X_test, y_train, y_test


def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    if random_state is not None:
        random.seed(random_state)
    X_train_folds = []
    X_test_folds = []
    length = len(X)
    test_spaces_left = length
    splits_left = n_splits+1
    test_index = 0
    window = list(range(length)) * 2
    while splits_left > 1:
        splits_left -= 1
        split = math.ceil(test_spaces_left / splits_left)
        X_test_folds.append([window[i]
                             for i in range(test_index, test_index+split)])
        test_range = range(test_index, test_index+split)
        test_index = test_index+split
        val = [i for i in range(length) if i not in test_range]
        X_train_folds.append(val)
        test_spaces_left -= split
        window = window * 2
    return X_train_folds, X_test_folds


def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
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
    total_folds = [[] for _ in range(n_splits)]
    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]

    # Group axes with group_by
    groupedList = myutils.group_by(X, y)
    # Create a pointer
    curr = 0
    # Iterate through outter list
    for group in groupedList:
        # Iterate through inner list (within individual elements in grouped list)
        for i in group:
            # Set pointer to current state + 1 mod n_splits
            curr = (curr + 1) % n_splits
            total_folds[curr].append(i)

    # New pointer
    curr = 0
    for j in range(n_splits):
        # Enumerate through fold list to get index positions
        for i, fold in enumerate(total_folds):
            if(i != j):
                for val in fold:
                    X_train_folds[curr].append(val)
            else:
                X_test_folds[curr] = fold

        curr += 1

    return X_train_folds, X_test_folds


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
    matrix = []
    length = len(labels)
    for i in range(length):
        matrix.append([0]*length)
        for j in range(length):
            for t, p in zip(y_true, y_pred):
                if (t, p) == (labels[i], labels[j]):
                    matrix[-1][j] += 1
    return matrix
