from statistics import mean
import math
import random
import itertools
from collections import Counter
from operator import itemgetter
import copy
import re


def frequency_count(table, index):
    """Partitions the table by the attribute index and returns list of tables"""
    frequency = []
    keys = list(set([row[index] for row in table]))
    frequency = [[key, 0] for key in keys]
    for i, key in enumerate(keys):
        for row in table:
            if key == row[index]:
                frequency[i][-1] += 1
    return frequency
    

def slope(xs, ys):
    xs = [x[0] for x in xs]
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_to_y_mean = mean([i * j for i, j in zip(xs, ys)])
    x_to_x_mean = mean([x*x for x in xs])
    m = ((x_mean * y_mean) - x_to_y_mean) / ((x_mean * x_mean) - x_to_x_mean)
    b = mean(ys) - (m * mean(xs))
    return m, b


def squared_error(ys_orig, ys_line):
    result = [(i - j) ** 2 for i, j in zip(ys_orig, ys_line)]
    return round(sum(result), 2)


def coefficient_correlation(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return round(1 - (squared_error_regr / squared_error_y_mean), 2)


def euclidean_distance(instance1, instance2, length):
    if type(instance1[0]) == str and type(instance2[0]) == str:
        for x in range(length):
            if instance1[x] == instance2[x]:
                return 0
            else:
                return 1
    else:
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


def min_max_scale(X):
    """Apply min-max scale to matrix of features"""
    # Scale Formula (x - min(xs)) / ((max(xs) - min(xs)) * 1.0)
    new_X = []
    for i in range(len(X[0])):
        col = [x[i] for x in X]
        x_min = min(col)
        x_max = max(col)
        scaled_x1 = [round(((x - x_min) / (x_max - x_min)) * 1.0, 4)
                     for x in col]
        new_X.append(scaled_x1)
    new_X = [list(x) for x in zip(*new_X)]
    return new_X


def accuracy(y_pred, y_true):
    """Calculates and returns accuracy and error between actual and predicted values"""
    correct = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            correct += 1
    accuracy = round((correct/len(y_pred)), 2)
    return accuracy


def group_by(X, y):
    """Splits the data into groups which are split on the axes"""
    labels = []
    counts = []

    # Append labels and get the count of how many labels there are
    for label in y:
        if label not in labels:
            labels.append(label)
            counts.append(1)
        elif label in labels:
            index = labels.index(label)
            counts[index] += 1
    ran = len(y)//len(labels)

    # Store grouped variables in list
    groupedList = []
    # Store 2d labels
    labels2 = []

    # Iterate through y for the length of y
    for x in range(len(y)):
        # For each element in y at the current index, check if element in labels2
        if y[x] not in labels2:
            # If not, append the element to labels2
            labels2.append(x)
            # Then check in labels...
            for i in range(len(labels)):
                # to check if labels match
                for lbl in range(len(y)):
                    if y[lbl] == labels[i]:
                        # append the label to grouped
                        groupedList.append(lbl)
        break
    # Return grouped list
    print(groupedList)

    # Create the inner list
    twoDList = []
    idx = 0
    for row in range(ran):
        inner_list = []
        for col in range(len(labels)):
            if idx != len(groupedList):
                inner_list.append(groupedList[idx])
                idx = idx + 1
        twoDList.append(inner_list)
    print(twoDList)

    # Returns a 2D list of grouppings
    return twoDList


def compute_holdout_partitions(table):
    # randomize the table
    randomized = table[:]  # copy the table
    random.shuffle(randomized)
    n = len(table)
    for i in range(0, n, 10):
        # pick an index to swap
        j = random.randrange(0, n)  # random int in [0,n)
        randomized[i], randomized[j] = randomized[j], randomized[i]
    # return train and test sets
    split_index = int(2/3 * n)  # 2/3 of randomized table is train, 1/3 is test
    return randomized[0:split_index], randomized[split_index:]


def gaussian(x, mean, sdev):
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second


def entropy(y):
    """Returns entropy for the given distribution of classes"""
    length = len(y)
    classes_ratio = [value/length for key, value in Counter(y).items()]
    entropy = 0
    for ratio in classes_ratio:
        log_result = -ratio * math.log2(ratio)
        entropy += log_result
    return entropy


def best_split(training_set, available_attributes):
    """Finds best attributes to perform split"""
    y_train = [y[-1] for y in training_set]
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
    # this is a group by split_attribute's domain, not by
    # the values of this attribute in instances
    # example: if split_attribute is "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0
    # lets build a dictionary
    partitions = {} # key (attribute value): value (list of instances with this attribute value)
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions


def all_same_class(partition):
    """Returns True or False based on whether partition contains same class in all instances or not."""
    label = None
    first_iteration = True
    for row in partition:
        if first_iteration:
            first_iteration = False
            label = row[-1]
        if row[-1] != label:
            return False
    return True


def majority_vote(partition):
    """Returns class with most occurrences in partition"""
    class_count = {}
    for row in partition:
        if row[-1] not in class_count:
            class_count[row[-1]] = 0
        class_count[row[-1]] += 1
    class_with_majority = max(class_count.items(), key=itemgetter(1))[0]
    return class_with_majority


def tdidt(current_instances, available_attributes, attribute_domains, header):
    # basic approach (uses recursion!!):
    # select an attribute to split on
    split_attribute = best_split(current_instances, available_attributes)
    available_attributes.remove(split_attribute)
    # cannot split on the same attribute twice in a branch
    # recall: python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domains, header)
    classes_count = {}
    total_instances = 0
    for key, values in partitions.items():
        total_instances += len(values)
        for val in values:
            if val:
                if val[-1] not in classes_count:
                    classes_count[val[-1]] = 0
                classes_count[val[-1]] += 1
    majority_class = max(classes_count.items(), key=itemgetter(1)) # for case 3
    m_vote, majority_count = majority_class[0], majority_class[1]

    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        value_subtree = ["Value", attribute_value]
        # TODO: appending leaf nodes and subtrees appropriately to value_subtree
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            value_subtree.append(["Leaf", partition[0][-1], len(partition), total_instances])
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            majority_class = majority_vote(partition)
            value_subtree.append(["Leaf", majority_class, len(partition), total_instances])
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            tree = ["Leaf", m_vote, majority_count, total_instances]
            return tree
        else: # all base cases are false... recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains, header)
            # need to append subtree to value_subtree and appropriately append value subtree to tree
            value_subtree.append(subtree)
        tree.append(value_subtree)
    return tree


def classifySample(instance, tree):
    """Classifies sample with given class labels"""
    y = None
    if tree[0] == 'Attribute':
        y = classifySample(instance, tree[2:])
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if tree[i][1] in instance:
                y = classifySample(instance, tree[i][2])
                break
    if tree[0] == 'Leaf':
        return tree[1]
    return y


def extractRules(tree, rules, chain, previous_value, class_name):
    """Extracts rules from the constructed decision tree."""
    if tree[0] == 'Attribute':
        if chain:
            chain += ' AND' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        else:
            chain = 'IF' + ' ' + str(tree[1]) + ' ' + '==' + ' '
        rules = extractRules(tree[2:], rules, chain, previous_value, class_name)
    if tree[0][0] == 'Value':
        for i in range(len(tree)):
            if previous_value and previous_value == chain[-len(previous_value):]:
                length = len(previous_value)
                chain = chain[:-length] + ' '
            chain += str(tree[i][1])
            previous_value = str(tree[i][1])
            rules = extractRules(tree[i][2], rules, chain, previous_value, class_name)
    if tree[0] == 'Leaf':
        chain += ' THEN' + ' ' + class_name + ' ' + '=' + ' ' + str(tree[1])
        chain = re.sub(' +', ' ', chain)
        rules.append(chain)
    return rules




