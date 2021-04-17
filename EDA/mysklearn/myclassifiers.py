import mysklearn.myutils as myutils
from collections import Counter
import random


class MySimpleLinearRegressor:
	"""Represents a simple linear regressor.

	Attributes:
		slope(float): m in the equation y = mx + b
		intercept(float): b in the equation y = mx + b

	Notes:
		Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
		Terminology: instance = sample = row and attribute = feature = column
	"""

	def __init__(self, slope=None, intercept=None):
		"""Initializer for MySimpleLinearRegressor.

		Args:
			slope(float): m in the equation y = mx + b (None if to be computed with fit())
			intercept(float): b in the equation y = mx + b (None if to be computed with fit())
		"""
		self.slope = slope
		self.intercept = intercept

	def fit(self, X_train, y_train):
		"""Fits a simple linear regression line to X_train and y_train.

		Args:
			X_train(list of list of numeric vals): The list of training samples
				The shape of X_train is (n_train_samples, n_features)
				Note that n_features for simple regression is 1, so each sample is a list 
					with one element e.g. [[0], [1], [2]]
			y_train(list of numeric vals): The target y values (parallel to X_train) 
				The shape of y_train is n_train_samples
		"""
		# TODO: copy your solution from PA4 here
		self.slope, self.intercept = myutils.slope(X_train, y_train)

	def predict(self, X_test):
		"""Makes predictions for test samples in X_test.

		Args:
			X_test(list of list of numeric vals): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)
				Note that n_features for simple regression is 1, so each sample is a list 
					with one element e.g. [[0], [1], [2]]

		Returns:
			y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
		"""
		# TODO: copy your solution from PA4 here
		predictions = [round((self.slope * x[0]) + self.intercept, 1)
					   for x in X_test]
		return predictions


class MyKNeighborsClassifier:
	"""Represents a simple k nearest neighbors classifier.

	Attributes:
		n_neighbors(int): number of k neighbors
		X_train(list of list of numeric vals): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
		y_train(list of obj): The target y values (parallel to X_train). 
			The shape of y_train is n_samples

	Notes:
		Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
		Terminology: instance = sample = row and attribute = feature = column
		Assumes data has been properly normalized before use.
	"""

	def __init__(self, n_neighbors=3):
		"""Initializer for MyKNeighborsClassifier.

		Args:
			n_neighbors(int): number of k neighbors
		"""
		self.n_neighbors = n_neighbors
		self.X_train = None
		self.y_train = None

	def fit(self, X_train, y_train):
		"""Fits a kNN classifier to X_train and y_train.

		Args:
			X_train(list of list of numeric vals): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
			y_train(list of obj): The target y values (parallel to X_train)
				The shape of y_train is n_train_samples

		Notes:
			Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
		"""
		self.X_train = X_train
		self.y_train = y_train

	def kneighbors(self, X_test):
		"""Determines the k closes neighbors of each test instance.

		Args:
			X_test(list of list of numeric vals): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			distances(list of list of float): 2D list of k nearest neighbor distances 
				for each instance in X_test
			neighbor_indices(list of list of int): 2D list of k nearest neighbor
				indices in X_train (parallel to distances)
		"""
		# TODO: copy your solution from PA4 here
		distances = []
		neighbor_indices = []
		for x_test in X_test:
			k_distances = []
			for index, x_train in enumerate(self.X_train):
				dist = myutils.euclidean_distance(
					x_train, x_test, len(x_train))
				k_distances.append((index, dist))
			k_distances = sorted(k_distances, key=lambda x: x[1])[
				:self.n_neighbors]
			indices = [i[0] for i in k_distances]
			sorted_dist = [d[1] for d in k_distances]
			neighbor_indices.append(indices)
			distances.append(sorted_dist)
		return distances, neighbor_indices

	def predict(self, X_test):
		"""Makes predictions for test instances in X_test.

		Args:
			X_test(list of list of numeric vals): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			y_predicted(list of obj): The predicted target y values (parallel to X_test)
		"""
		# TODO: copy your solution from PA4 here
		distances, neighbor_indices = self.kneighbors(X_test)
		y_predicted = []
		for i in range(len(X_test)):
			labels = [self.y_train[j] for j in neighbor_indices[i]]
			most_common = Counter(labels).most_common(1)[0][0]
			y_predicted.append(most_common)
		return y_predicted


class MyNaiveBayesClassifier:
	"""Represents a Naive Bayes classifier.

	Attributes:
		X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
		y_train(list of obj): The target y values (parallel to X_train). 
			The shape of y_train is n_samples
		priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
			label in the training set.
		posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
			attribute value/label pair in the training set.

	Notes:
		Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
		Terminology: instance = sample = row and attribute = feature = column
	"""

	def __init__(self):
		"""Initializer for MyNaiveBayesClassifier.

		"""
		self.X_train = None
		self.y_train = None
		self.priors = None
		self.posteriors = None

	def fit(self, X_train, y_train):
		"""Fits a Naive Bayes classifier to X_train and y_train.

		Args:
			X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
			y_train(list of obj): The target y values (parallel to X_train)
				The shape of y_train is n_train_samples

		Notes:
			Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
				and the posterior probabilities for the training data.
			You are free to choose the most appropriate data structures for storing the priors
				and posteriors.
		"""
		self.priors = {}
		self.posteriors = {}

		labels_count = {}
		for label in y_train:
			if label not in labels_count:
				labels_count[label] = 0
			labels_count[label] += 1

		unique_attributes_values = [[] for i in range(len(X_train[0]))]

		for row in X_train:
			for i, val in enumerate(row):
				if val not in unique_attributes_values[i]:
					unique_attributes_values[i].append(val)

		# Getting Prior Probabilities

		length = len(X_train)
		for key, value in labels_count.items():
			self.priors[key] = round(value / length, 2)

		# Getting Posterior Probabilities

		for label in labels_count:
			self.posteriors[label] = {}
			for attr_index, attribute in enumerate(unique_attributes_values):
				self.posteriors[label][attr_index] = {}
				for val_index, val in enumerate(attribute):
					matches = 0
					for i, row in enumerate(X_train):
						if val == row[attr_index] and y_train[i] == label:
							matches += 1
					self.posteriors[label][attr_index][val] = round(
						matches / labels_count[label], 2)

	def predict(self, X_test):
		"""Makes predictions for test instances in X_test.

		Args:
			X_test(list of list of obj): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			y_predicted(list of obj): The predicted target y values (parallel to X_test)
		"""
		y_predicted = []
		for row in X_test:
			probabilities = []
			for label, columns in self.posteriors.items():
				result = self.priors[label]
				col_keys = list(columns.keys())
				for i, value in enumerate(row):
					try:
						result *= columns[col_keys[i]][value]
					except:
						result *= columns[float(col_keys[i])][value]
				probabilities.append((label, result))
			prediction = sorted(probabilities, reverse=True, key=lambda x: x[1])[
				0][0]  # getting label with highest probability value
			y_predicted.append(prediction)
		return y_predicted


class MyZeroRClassifier:
	"""Represents a Zero R classifier.

	Attributes:
		X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
		y_train(list of obj): The target y values (parallel to X_train). 
			The shape of y_train is n_samples
	"""

	def __init__(self):
		"""Initializer for MyZeroRClassifier.

		"""
		self.X_train = None
		self.y_train = None

	def fit(self, X_train, y_train):
		"""Fits a Zero R classifier to X_train and y_train.

		Args:
			X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
			y_train(list of obj): The target y values (parallel to X_train)
				The shape of y_train is n_train_samples
		"""
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		"""Makes predictions for test instances in X_test.

		Args:
			X_test(list of list of obj): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			y_predicted(list of obj): The predicted target y values (parallel to X_test)
		"""
		y_predicted = []
		most_common_label = Counter(self.y_train).most_common(1)[0][0]
		for i in range(len(X_test)):
			y_predicted.append(most_common_label)
		return y_predicted


class MyRandomClassifier:
	"""Represents a Random classifier.

	Attributes:
		X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
		y_train(list of obj): The target y values (parallel to X_train). 
			The shape of y_train is n_samples
	"""

	def __init__(self):
		"""Initializer for MyRandomClassifier.

		"""
		self.X_train = None
		self.y_train = None
		self.weights = None
		self.labels = None

	def fit(self, X_train, y_train):
		"""Fits a Random Classifier to X_train and y_train.

		Args:
			X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
			y_train(list of obj): The target y values (parallel to X_train)
				The shape of y_train is n_train_samples
		"""
		self.X_train = X_train
		self.y_train = y_train
		frequencies = Counter(y_train)
		self.labels = list(frequencies.keys())
		labels_count = list(frequencies.values())
		self.weights = []
		label_length = len(y_train)
		for c in labels_count:
			weight = round((c * 100) / label_length)
			self.weights.append(weight)

	def predict(self, X_test):
		"""Makes predictions for test instances in X_test.

		Args:
			X_test(list of list of obj): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			y_predicted(list of obj): The predicted target y values (parallel to X_test)
		"""
		y_predicted = []
		for row in X_test:
			# random value generation based on labels' weight
			prediction = random.choices(self.labels, weights=self.weights)[0]
			y_predicted.append(prediction)
		return y_predicted


class MyDecisionTreeClassifier:
	"""Represents a decision tree classifier.

	Attributes:
		X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
		y_train(list of obj): The target y values (parallel to X_train). 
			The shape of y_train is n_samples
		tree(nested list): The extracted tree model.

	Notes:
		Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
		Terminology: instance = sample = row and attribute = feature = column
	"""

	def __init__(self):
		"""Initializer for MyDecisionTreeClassifier.

		"""
		self.X_train = None
		self.y_train = None
		self.tree = None

	def fit(self, X_train, y_train):
		"""Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

		Args:
			X_train(list of list of obj): The list of training instances (samples). 
				The shape of X_train is (n_train_samples, n_features)
			y_train(list of obj): The target y values (parallel to X_train)
				The shape of y_train is n_train_samples

		Notes:
			Since TDIDT is an eager learning algorithm, this method builds a decision tree model
				from the training data.
			Build a decision tree using the nested list representation described in class.
			Store the tree in the tree attribute.
			Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
		"""
		self.X_train = X_train
		self.y_train = y_train
		header = ['att' + str(i) for i in range(len(self.X_train[0]))]  # Computing headers
		del self.X_train[0]
		del self.y_train[0]

		# Computing attribute domains
		attribute_domains = {}
		for i, h in enumerate(header):
			attribute_domains[h] = []
			for x in self.X_train:
				if x[i] not in attribute_domains[h]:
					attribute_domains[h].append(x[i])

		for k, v in attribute_domains.items():
			attribute_domains[k] = sorted(v)

		training_set = [self.X_train[i] + [self.y_train[i]] for i in range(len(self.X_train))]
		# initial call to tdidt current instances is the whole table (train)
		available_attributes = header.copy() # python is pass object reference
		self.tree = myutils.tdidt(training_set, available_attributes, attribute_domains, header)

	def predict(self, X_test):
		"""Makes predictions for test instances in X_test.

		Args:
			X_test(list of list of obj): The list of testing samples
				The shape of X_test is (n_test_samples, n_features)

		Returns:
			y_predicted(list of obj): The predicted target y values (parallel to X_test)
		"""
		y_predicted = []
		for instance in X_test:
			y_pred = myutils.classifySample(instance, self.tree)
			y_predicted.append(y_pred)
		return y_predicted

	def print_decision_rules(self, attribute_names=None, class_name="class"):
		"""Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

		Args:
			attribute_names(list of str or None): A list of attribute names to use in the decision rules
				(None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
			class_name(str): A string to use for the class name in the decision rules
				("class" if a string is not provided and the default name "class" should be used).
		"""
		rules = myutils.extractRules(tree=self.tree, rules=[], chain='' , previous_value='', class_name=class_name)
		for rule in rules:
			print(rule)

	# BONUS METHOD
	def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
		"""BONUS: Visualizes a tree via the open source Graphviz graph visualization package and its DOT graph language (produces .dot and .pdf files).

		Args:
			dot_fname(str): The name of the .dot output file.
			pdf_fname(str): The name of the .pdf output file generated from the .dot file.
			attribute_names(list of str or None): A list of attribute names to use in the decision rules
				(None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).

		Notes: 
			Graphviz: https://graphviz.org/
			DOT language: https://graphviz.org/doc/info/lang.html
			You will need to install graphviz in the Docker container as shown in class to complete this method.
		"""
		import pygraphviz as pgv
		G = pgv.AGraph(self.tree)
		G.write('tree_graph.dot')
