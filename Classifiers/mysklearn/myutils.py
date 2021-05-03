"""
Luke Mason & Karsen Hansen
CPSC 322: Final Project
Apr 21 2021
myutils.py
"""
import math
import operator
import copy
import random
from functools import reduce
from tabulate import tabulate
from operator import itemgetter
from collections import Counter
import itertools
import re
from mysklearn.myclassifiers import MyDecisionTreeClassifier
from mysklearn import myevaluation

def compute_best_fit(x, y):
    """ Function to compute the slope and intercept of a best fit line for two data sets
        
        Params -
                x: one set of data
                y: the other set of data
        
        Return - 
                m: the slope of the best fit line
                b: the intercept of the best fit line
    """
    # Get the mean values
    x_mean = sum(x)/len(x)
    y_mean = sum(y)/len(y)

    # For each value in the x and y data set, sum the differences of those values against the mean of their set, 
    # then divide by the sum of the x mean subtracted from the x data values 
    m = sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))]) / sum([(x[i] - x_mean) ** 2 for i in range(len(x))])
    b = y_mean - m*x_mean # Trivially get the intercept using y = mx + b where y and x are the mean values of the input sets

    return m, b

def group_by(table, index):
    """ Function to get the subtables of a grouped by table on the index passed in
        
        Params -
                table: table to get the data from
                index: index of the column to group by
        
        Return - 
                group_names: the names of the groupped column
                group_subtables: the data that fits into each group
    """
    col = get_column(table, index) # Get the values for the column to group by
    col_index = index # Set the index from the passed in variable
    
    # Get the unique values from our column and init the subtables lists
    group_names = sorted(list(set(col)))
    group_subtables = [[] for _ in group_names]
    
    # Loop through the table to create subtables
    for i in range(len(table)):
        group_by_value = table[i][col_index] # Get the value we will group by
        group_index = group_names.index(group_by_value) # Get the index of the subtable we wil add the value to
        group_subtables[group_index].append(i) # Add the value
    
    return group_names, group_subtables

def get_column(table, index):
    """ Function to get the column of a table given the index of the column
        
        Params -
                table: table to get the data from
                index: index of the column to group by
        
        Return - 
                col: the data within the associated column
    """
    # Get the index and init the return list
    col_index = index
    col = []

    # Loop through the table and append values
    for row in table: 
        # Ignore missing values ("NA")
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def euclidean_distance(instance1, instance2, length):
    """ Function to get the euclidean distance between two instances given a length
        
        Params -
                instance1: the first instance to get the distance from
                instance2: the second instance to get the distance to
                length: the length of the parallel lists
        
        Return - 
                the value of the euclidean distance
    """
    distance = 0 # Init distance

    # Loop through the parallel lists
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2) # Sum distance values
    return math.sqrt(distance)

def kneighbors_helper(scaled_train, scaled_test, n_neighbors):
    """
        Helper function for kneighbors distances and indices
        params:
                scaled_train - the scaled train values
                scaled_test - the scaled test values
                n_neighbors - the number of neighbors
        returns:
                the distances and indices for kneighbors
    """
    # Make two copies on the params
    scaled_train_copy = copy.deepcopy(scaled_train)
    scaled_test_copy = copy.deepcopy(scaled_test)

    # Iterate through the scaled train values
    for i, instance in enumerate(scaled_train_copy):
        # Append the index and distances to the instance list
        instance.append(i)
        dist = euclidean_distance(instance[:-1], scaled_test_copy, len(scaled_test_copy))
        instance.append(dist)
    
    # Sort the scaled train values
    train_sorted = sorted(scaled_train_copy, key=operator.itemgetter(-1))
 
    # Get the beginning of the sorted training list
    top_k = train_sorted[:n_neighbors]

    # Init return lists
    distances = []
    indices = []

    # Loop through the beginning kneighbors
    for row in top_k:
        # Append the distance and index associated with the row
        distances.append(row[-1])
        indices.append(row[-2])
    
    return distances, indices

def scale(vals, test_vals):
    """
        ** Note ** - see jupyter notebook, this is the function I received help from 
                     Armando Valdez on
        Function to scale the values of the passed in parameter
        params:
                values - the values to be scaled
                test_values - the values to iterate over when scaling
        returns:
                the label that appears most often in the paramter list
    """
    # Init lists
    return_list = []
    max_vals = []
    min_vals = []

    # Iterate through the list at the beginning of the list
    for i in range(len(vals[0])):
        # Get the max and min values in the list and append
        max_vals.append(max([val[i] for val in vals]))
        min_vals.append(min([val[i] for val in vals]))

    # Iterate through the values
    for row in vals:
        # Get the current value
        curr = []

        # Iterate through the current row
        for i in range(len(row)):
            # Get the scaled x value and append it to the list
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        
        # Append the scaled values for the row
        return_list.append(curr)
    
    # Iterate through the test values
    for row in test_vals:
        # Get the current test value
        curr = []

        # Iterate through the row in the test value list of lists
        for i in range(len(row)):
            # Get the scaled test value and append it to the list
            curr.append((row[i]-min_vals[i])/(max_vals[i]-min_vals[i]))
        
        # Append the scaled values for the row
        return_list.append(curr)
    
    # Return the two lists scaled
    return return_list[:len(vals)], return_list[len(vals):]

def get_max_label(labels):
    """
        Function to get the label that appears most freqency in a labels list
        params:
                labels - the list of labels
        returns:
                the label that appears most often in the paramter list
    """
    # Init unique list
    set_of_labels = []

    # Loop through labels and get set of labels
    for label in labels:
        if label not in set_of_labels:
            set_of_labels.append(label)

    # Init counts of parallel list to 0
    counts = []
    for _ in range(len(set_of_labels)):
        counts.append(0)
    
    # Loop through set of labels and increment the counts
    for i, set_label in enumerate(set_of_labels):
        for label in labels:
            if set_label == label:
                counts[i] += 1 # increment

    # Default max label count and label
    max_count = 0
    max_label = ""

    # Loop through counts
    for i, count in enumerate(counts):
        if count > max_count:
            max_label = set_of_labels[i]
 
    return max_label

def get_rand_rows(table, num_rows):
    """
        Function to get a random number of rows from a passed in table
        params:
                table - the table to pull rows from
                num_rows - the number of random rows to pull
    """
    # Init list
    rand_rows = []

    # Iterate the number of rows times
    for _ in range(num_rows):
        # Appenda  random row
        rand_index = random.randint(0, len(table.data)) - 1
        rand_rows.append(table.data[rand_index])
    return rand_rows

def get_rating_mpg(val):
    """
        Function to bin the associated mpg value with a rating
        params:
                val - the value to be binned
    """
    if val <= 13:
        return 1
    elif val == 14:
        return 2
    elif val < 16:
        return 3
    elif val < 19:
        return 4
    elif val < 23:
        return 5
    elif val < 26:
        return 6
    elif val < 30:
        return 7
    elif val < 36:
        return 8
    elif val < 44:
        return 9
    else:
        return 10

def get_values_from_folds(x_vals, y_vals, train_folds, test_folds):
    """
        Function to pull the values from the indexed folds from the value lists
        params:
                x_vals - x data set with values
                y_vals - parallel list that is the y data set with values
                train_folds - the determined training folds to pull values from x and y using indices
                test_folds - the determined testing folds to pull values from x and y using indices
    """
    # Init lists
    x_train = []
    y_train = []

    # Iterate through training folds
    for row in train_folds:
        for i in row:
            # Append values to lists
            x_train.append(x_vals[i])
            y_train.append(y_vals[i])

    # Init lists
    x_test = []
    y_test = []

    # Iterate through testing folds
    for row in test_folds:
        for i in row:
            # Append values to lists
            x_test.append(x_vals[i])
            y_test.append(y_vals[i])

    return x_train, y_train, x_test, y_test

def accuracy_count(predicted, expected):
    """
        Function to count the amount of predicted values that were accurate
        params:
                predicted - A list of predicted values
                expected - A parallel list of actual values
    """
    # Default count to 0
    count = 0

    # Double check parallel arrays
    assert len(predicted) == len(expected)

    # Loop through list
    for i in predicted:
        # Check if equivalent and increment count
        if predicted[i] == expected[i]:
            count += 1
    return count

def calc_matrix_stats(matrix, titanic=False):
    """
        Function to add necessary stats to a confusion matrix
        params:
                matrix - the matrix to compute the stats over
    """
    complete_matrix = []

    # Loop through the enumerated matrix
    for i in range(len(matrix)):
        row = []
        if titanic:
            if i == 0:
                row.append("Yes")
            else:
                row.append("No")
        else:
            row.append(i + 1)
        sum = 0
        for j in range(len(matrix[i])):
            row.append(matrix[i][j])
            sum = sum + matrix[i][j]
        row.append(sum)
        if sum == 0:
            row.append(0)
        else:
            row.append(round(row[i+1]/row[-1]*100,2))
        complete_matrix.append(row)

    return complete_matrix


def print_tabulate(table, header):
    """
    Function to pretty print the table
    params:
            table - the table to be printed
            headers - the headers associated with the table
    """
    print(tabulate(table, header, tablefmt="rst"))

def get_priors(y_train):
    """
    Function to get the priors labels for Naive Bayes
    params:
            y_train - the labels for the training set
    return:
            a dictionary holding the prior labels and their associated probability
    """
    unique = [] # list to hold the unique labels
    counts = [] # list to hold the counts for the labels

    # Loop through labels
    for label in y_train:
        # Check if unique
        if label in unique:
            # Get the index for the label in unique
            index = unique.index(label)
            counts[index] = counts[index] + 1 # increment count
        else:
            # Add the label to unique and initialize the count
            unique.append(label)
            counts.append(1)
    
    denom = len(y_train) # Get the denominator for the priors calcs
    priors_dict = {}
    
    # Loop through the unique labels
    for i in range(len(unique)):
        label = unique[i]
        priors_dict[label] = counts[i]/denom # Prior calculation
    
    return priors_dict

def get_posteriors(X_train, y_train, priors):
    """
    Function to get the posteriors
    params:
            X_train - the instances to train on
            y_train - the labels to train on 
            priors - the dictionary holding the priors
    return:
            a dictionary holding the posterior labels and their associated probability
    """
    # e.g. X_train = [
    #                   [1, 5], 
    #                   [2, 6], 
    #                   [1, 5], 
    #                   [1, 5], 
    #                   [1, 6],
    #                   [2, 6],
    #                   [1, 5],
    #                   [1, 6]
    #                           ]

    # e.g. y_train = [
    #                   yes, 
    #                   yes, 
    #                   no, 
    #                   no, 
    #                   yes,
    #                   no,
    #                   yes,
    #                   yes
    #                           ]

    # e.g. priors = {"yes": 5/8, "no": 3/8}
    
    posteriors = {}

    # Loop through the priors to build up dictionary structure
    for k, _ in priors.items():
        posteriors[k] = {}
        for i in range(len(X_train[0])):
            posteriors[k][i] = {}
    
    # e.g posteriors = {"yes": {"0": {}, "1": {}}, "no": {"0": {}, "1": {}}}

    # Loop through X_train instances 
    for j in range(len(X_train)):
        for i in range(len(X_train[j])):
            prior_label = y_train[j] # Get the prior label to index into posteriors dictionary
            posterior_label = X_train[j][i] # Get the posterior label
            denom = priors[prior_label] * len(y_train)

            if posterior_label in posteriors[prior_label][i]:
                # Adjust the posteriors value that is already set
                posteriors[prior_label][i][posterior_label] = ((posteriors[prior_label][i][posterior_label] * denom) + 1) / denom
            else:
                # Set the posteriors value for the first time
                posteriors[prior_label][i][posterior_label] = 1 / denom
        
    
    # e.g posteriors = {"yes": {"0": {"1": 4/5, "2": 1/5}, "1": {"5": 2/5, "6": 3/5}}, "no": {"0": {"1": 2/3, "2": 1/3}, "1": {"5": 2/3, "6": 1/3}}}

    return posteriors

def multiply(a, b):
    """
    Function to multiply two values
    params:
            a - one operand
            b - the second operand
    return:
            the multiplication value of a*b
    """
    return a*b

def compute_probs(test, priors, posteriors):
    """
    Function to compute the probabilities of a test set
    params:
            test - the test set
            priors - the priors dictionary
            posteriors - the posteriors dictionary
    return:
            a return dictionary with the probalities for the test instances
    """
    return_dictionary = {}

    # Loop through the priors
    for k, v in priors.items():
        prior = v
        dictionary = posteriors[k] # Get the posteriors dictionary
        probs = []
        probs.append(prior) # Append the prior probability

        # Loop through the test
        for i in range(len(test)):
            if test[i] in dictionary[i]:  
                # Append the probability value
                probs.append(dictionary[i][test[i]])
            else:
                # Not in the dictionary, append a probability of 0
                probs.append(0)

        # Reduce the list by multiplying all values
        probability = reduce(multiply, probs)
        return_dictionary[k] = probability # Set the dictionary
    
    return return_dictionary

def predict_from(probs_dictionary):
    """
    Function to make a prediction from a dictionary
    params:
            probs_dictionary: the dictionary holding the probabilities
    return:
            the prediction label
    """
    # Init
    max = 0 
    prediction = ""

    # Loop through probabilities and check maxes
    for k, v, in probs_dictionary.items():
        if v >= max:
            prediction = k
            max = v
    
    # Return the prediction with the highest probability
    return prediction

def get_common_class(y_train):
    """
    Function to get the most common class label in y_train
    params:
            y_train: the list of labels to train on
    return:
            the most common label label
    """
    # Get the unique labels and their counts
    labels = []
    counts = []
    for label in y_train:
        if label in labels:
            idx = labels.index(label)
            counts[idx] = counts[idx] + 1
        else:
            labels.append(label)
            counts.append(0)
    
    # Get the index of the label with the most appearances
    max_idx = counts.index(max(counts))

    return labels[max_idx]

def reduce_data(data, header, atts):
    """
    Function to get rid of the columns that don't matter to the user
    params:
            data: the table to reduce
            header: the header of the expanded table
            atts: the attributes to reduce down to
    return:
            the smaller table
    """
     # List of the indices for columns we want and table of reduced data
    idxs = []
    reduced_data = []

    # Loop through the attributes
    for name in atts:
        idxs.append(header.index(name)) # Append the index of an attribute we wanna keep
    
    # Loop through data and reduce
    for row in data:
        reduced_row = []
        for idx in idxs:
            reduced_row.append(row[idx])
        
        reduced_data.append(reduced_row)
    
    return reduced_data

def split_x_y_train(data):
    """
    Function to split the data into x and y with y being the last column in a table
    params:
            data: the table to split
    return:
            the x_train data list[:-1]
            the y_train data list[-1]
    """
    x_train = []
    y_train = []
    for row in data:
        x_train.append(row[:-1])
        y_train.append(row[-1])
    
    return x_train, y_train

def convert(value):
    """
    Helper function to convert the weight values
    params:
            value: the value to convert
    return:
            the binned weight value
    """
    if value >= 3500:
        return 5
    elif value >= 3000:
        return 4
    elif value >= 2500:
        return 3
    elif value >= 2000:
        return 2
    else:
        return 1

def convert_weight(auto_data_reduced, idx):
    """
    Function to convert the weight values
    params:
            auto_data_reduced: auto data that will bin the weight values
            idx: the index of the column that weight is in
    return:
            the adjusted data
    """
    training_data = []
    for row in auto_data_reduced:
        row_copy = copy.deepcopy(row)
        row_copy[idx] = convert(row_copy[idx])
        training_data.append(row_copy)

    return training_data


def get_rand_rows_table(data, num_rows):
    """
        Function to get a random number of rows from a passed in table
        params:
                data - the table to pull rows from
                num_rows - the number of random rows to pull
    """
    # Init list
    rand_X = []
    rand_y = []

    # Iterate the number of rows times
    for _ in range(num_rows):
        # Appenda random row
        rand_index = random.randint(0, len(data)) - 1
        rand_rows = data[rand_index]
        rand_X.append(rand_rows[:-1])
        rand_y.append(rand_rows[-1])
    return rand_X, rand_y

#---------------------------------------------------------------------
#-------------------------Decision Tree-------------------------------
#---------------------------------------------------------------------

def entropy(y):
    """ 
    Function to get the entropy value

    params: 
        y - the classifier list to gather entropies from

    return:
        the entropy of y
    """
    denom = len(y) # Get the denominator for the entropy calculation

    # Init lists
    classes_ratio = []
    distinct = []
    counts = []

    # Loop through classifier values
    for classifier in y:
        # Check if new classifier
        if classifier in distinct:
            # Not new, increment count
            distinct_idx = distinct.index(classifier)
            counts[distinct_idx] += 1
        else:
            # New, add to list and init count to 1
            distinct.append(classifier)
            counts.append(1)

    # Loop through distinct list
    for i in range(len(distinct)):
        # Calculate the ratio to be used with log
        classes_ratio.append(counts[i]/denom)

    # Default the entropy sum
    entropy = 0
    # Loop through ratios
    for ratio in classes_ratio:
        # Get entropy and add it
        individual_entropy = -ratio * math.log2(ratio)
        entropy += individual_entropy
    return entropy


def best_split(training_set, available_attributes):
    """Finds best attributes to perform split"""
    y_train = []
    for y in training_set:
        y_train.append(y[-1])

    entropy_start = entropy(y_train)
    total_instances = len(y_train)
    information_gains = {}
    for col in available_attributes:
        attribute_entropies = {}
        training_set = sorted(training_set, key=itemgetter(int(col[-1])))
        group_iterator = itertools.groupby(training_set, key=itemgetter(int(col[-1])))
        for key, group in group_iterator:
            y = [x[-1] for x in group]
            instances = len(y)
            e = entropy(y)
            attribute_entropies[key] = (instances, e)
        entropy_new = 0
        for key, value in attribute_entropies.items():
            entropy_new += (value[0] / total_instances) * value[1] # weighted entropies
        information_gains[int(col[-1])] = entropy_start - entropy_new
    best_col = sorted(information_gains.items(), key=itemgetter(1), reverse=True)[0][0]
    for col in available_attributes:
        if int(col[-1]) == best_col:
            best_col = col
    return best_col


def partition_instances(instances, split_attribute, attribute_domains, header):
    """
    Function to split the partitions

    params:
        instances - a list of instances to split into partitions
        split_attribute - the attribute to group the partitions on
        attribute_domains - the partitions
        header - the names of the attributes
    
    return:
        a dictionary with the partitions
    """
    # The domain is the value of the attribute domains dictionary
    attribute_domain = attribute_domains[split_attribute]

    # The index of the attribute is grabbed from the header
    attribute_index = header.index(split_attribute)

    # Build up dictionary
    partitions = {}
    for attribute_value in attribute_domain:
        # Default partitions list
        partitions[attribute_value] = []

        # Loop through instances
        for instance in instances:
            # Check if the instance is in the attribute partition
            if instance[attribute_index] == attribute_value:
                # It is, append the instance to the list in the dictionary
                partitions[attribute_value].append(instance)
    return partitions


def all_same_class(partition):
    """
    Function that tells whether the partition contains same class in all instances or not.

    params:
        partition - the partition that we want to know if it has all the same class label

    return:
        True if the partitions has all the same class label
        False if the partition has more than one class label
    """
    classifier = None # Default to None
    first_loop = True # Default to true

    # Loop through the partition
    for row in partition:
        # Check if this is the first time we are looping
        if first_loop:
            first_loop = False # It is no longer the first time we are looping
            classifier = row[-1] # Set the classifier for the rest of the partition

        # Check comparison with first classifier
        if row[-1] != classifier:
            # Not a match, not all same class
            return False
    return True


def majority_vote(partition):
    """
    Function to return class with most occurrences in partition
    
    params:
        partition - the partition with the classes we are trying to find the max of

    return:
        the class label that occurs the most
    """
    # Init parallel arrays to find max class label occurrance
    distinct = []
    counts = []

    # Loop through the partition
    for row in partition:
        # Check if in distinct
        if row[-1] in distinct:
            # Increment count after grabbing index
            distinct_idx = distinct.index(row[-1])
            counts[distinct_idx] += 1
        else:
            # Append new count and new label
            distinct.append(row[-1])
            counts.append(1)

    # Get the max count, its index, and its label to return
    max_classifier_count = max(counts)
    max_idx = counts.index(max_classifier_count)
    class_with_majority = distinct[max_idx]
    return class_with_majority

def get_classes_and_instances(partitions):
    """
    Function to get the counts of classsifiers and the number of instanecs per class

    params:
        partitions - The partitions by attribute
    
    return:
        classes_count - a dictionary by classifier with their associated counts
        total_instances - the number of instances within all partitions
    """
    classes_count = {}
    total_instances = 0
    for _, v in partitions.items():
        total_instances += len(v)
        for val in v:
            if val:
                if val[-1] not in classes_count:
                    classes_count[val[-1]] = 0
                classes_count[val[-1]] += 1
    return classes_count, total_instances

def tdidt(current_instances, available_attributes, attribute_domains, header):
    """
    Function to build up a decision tree recurssively

    params:
        current_instances - the current instances we are using to build the tree
        available_attributes - what attributes we are able to split on
        attribute_domains - the dictionary containing the possible values of an attribute
        header - the header of the table containing attribute names
    
    return:
        The decision tree completely built up
    """
    # Get the attribute to split on
    split_attribute = best_split(current_instances, available_attributes)
    available_attributes.remove(split_attribute) # Clear the selected attribute from the available attributes

    # Build up the sub stree starting with the split attribute
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    classes_count, total_instances = get_classes_and_instances(partitions) # Get the counts per classifier and the instances in that classifier

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        value_subtree = ["Value", attribute_value]

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            value_subtree.append(["Leaf", partition[0][-1], len(partition), total_instances])

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            majority_class = majority_vote(partition)
            value_subtree.append(["Leaf", majority_class, len(partition), total_instances])

        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            majority_class = max(classes_count.items(), key=itemgetter(1)) # for case 3
            m_vote, majority_count = majority_class[0], majority_class[1]
            tree = ["Leaf", m_vote, majority_count, total_instances]
            return tree

        # all base cases are false... recurse!!
        else: 
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree


def classifySample(instance, tree):
    """
    Classifies sample with given class labels

    params:
        instance - the instance being classified in the sample
        tree - the tree the instance is being classified from

    return:
        classifier - the classifier associated with the leaf node for the instance
    """
    classifier = None # default

    # Check if the top of the tree is an attribute
    if tree[0] == 'Attribute':
        # Recurse to get to the leaf
        classifier = classifySample(instance, tree[2:])
    # Check if beneath the attribute is a value node
    if tree[0][0] == 'Value':
        # Go through the tree
        for i in range(len(tree)):
            # Check if the value is in the instance
            if tree[i][1] in instance:
                # Grab the leaf node associated with the instace
                classifier = classifySample(instance, tree[i][2])
                break
    if tree[0] == 'Leaf':
        # We found the leaf node and can classify
        return tree[1]
    return classifier


def extractRules(tree, rules, stmt, previous_value, class_name):
    """
    Extracts rules from the constructed decision tree.

    params: 
        tree - the tree to extract rules from
        rules - the returned list of rules
        stmt - the statement for each rule being built up
        previous_value - the value on the previous pass of the rules
        class_name - the classifier to finish the rule with

    return:
        rules - a list of rules for the tree
    """
    if tree[0] == 'Attribute':
        if stmt:
            stmt += ' AND' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        else:
            stmt = 'IF' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        rules = extractRules(tree[2:], rules, stmt, previous_value, class_name)
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if previous_value and previous_value == stmt[-len(previous_value):]:
                length = len(previous_value)
                stmt = stmt[:-length] + ' '
            stmt += str(tree[i][1])
            previous_value = str(tree[i][1])
            rules = extractRules(tree[i][2], rules, stmt, previous_value, class_name)
    if tree[0] == 'Leaf':
        stmt += ' THEN' + ' ' + class_name + ' ' + '=' + ' ' + str(tree[1])
        stmt = re.sub(' +', ' ', stmt)
        rules.append(stmt)
    return rules

def remove_rows_from_data(col_names, table):
    for row in table.data:
        adjustment = 0
        for name in col_names:
            idx = table.column_names.index(name)
            del row[idx - adjustment]
            adjustment += 1

    for name in col_names:
            idx = table.column_names.index(name)
            del table.column_names[idx]

def get_friend_count(idx, table):
    for row in table.data:
        friend_string = str(row[idx])
        commaless = friend_string.replace(',', '')
        spaceless = commaless.replace(' ', '')
        friend_count = len(spaceless)/22
        row[idx] = friend_count

def get_useful_bin(val):
    if val < 10:
        return 1
    elif val < 30:
        return 2
    elif val < 60:
        return 3
    elif val < 100:
        return 4
    elif val < 300:
        return 5
    elif val < 600:
        return 6
    elif val < 900:
        return 7
    elif val < 1300:
        return 8
    elif val < 2000:
        return 9
    else:
        return 10

#---------------------------------------------------------------------
#-------------------------Random Forest-------------------------------
#---------------------------------------------------------------------

def compute_bootstrapped_sample(table):
    """ Function to get the bootstrapped sample from a table

    params: table
    return: the sample of random instances with replacement from the table
    """
    n = len(table)
    sample = []
    validation = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])

    for row in table:
        if row not in sample:
            validation.append(row)

    return sample, validation

def sort_forest(accuracy_list, forest):
    sorted_forest = []
    unsorted = accuracy_list
    for i in range(len(accuracy_list)):
        unsorted = accuracy_list[i:]
        min_value = min(unsorted)
        unsorted.remove(min_value)
        tree_index = accuracy_list.index(min_value)
        sorted_forest.insert(0, forest[tree_index])

    return sorted_forest

def calculate_accuracy(y_predict, y_test):
    
    if (len(y_predict) != 0):
        count = 0
        for i in range(len(y_predict)):
            if y_predict[i] == y_test[i]:
                count += 1
        return count/len(y_predict)
    else:
        return 0

def prune_forest(forest, M):
    return forest[:M]

def random_forest_generation(remainder_set, N, M):
    complete_forest = []
    accuracy_forest = []
    pruned_forest = []
    for _ in range(N):
        training, validation = compute_bootstrapped_sample(remainder_set)
        X_train, y_train = split_x_y_train(training)
        X_test, y_test = split_x_y_train(validation)
        myDT = MyDecisionTreeClassifier()
        myDT.fit(X_train, y_train)
        y_predict = myDT.predict(X_test)
        accuracy = calculate_accuracy(y_predict, y_test)
        accuracy_forest.append(accuracy)
        complete_forest.append(myDT)
    sorted_forest = sort_forest(accuracy_forest, complete_forest)
    pruned_forest = prune_forest(sorted_forest, M)

    return pruned_forest

def get_majority_vote(predictions):
    distinct = []
    votes = []
    for prediction in predictions:
        if prediction in distinct:
            vote_idx = distinct.index(prediction)
            votes[vote_idx] += 1
        else:
            distinct.append(prediction)
            votes.append(1)
    
    max_idx = votes.index(max(votes))
    return distinct[max_idx]

