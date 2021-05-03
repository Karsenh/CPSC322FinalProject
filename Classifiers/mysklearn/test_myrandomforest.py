"""
Luke Mason & Karsen Hansen
CPSC 322: Project
May 3 2021
test_myrandomforest.py
"""

import numpy as np
from mysklearn import myutils
from mysklearn import myevaluation
import math
from mysklearn.myclassifiers import MyRandomForestClassifier


def test_random_forest_fit():
    # interview dataset
    table = [
            ["Senior", "Java", "no", "no", "False"],
            ["Senior", "Java", "no", "yes", "False"],
            ["Mid", "Python", "no", "no", "True"],
            ["Junior", "Python", "no", "no", "True"],
            ["Junior", "R", "yes", "no", "True"],
            ["Junior", "R", "yes", "yes", "False"],
            ["Mid", "R", "yes", "yes", "True"],
            ["Senior", "Python", "no", "no", "False"],
            ["Senior", "R", "yes", "no", "True"],
            ["Junior", "Python", "yes", "no", "True"],
            ["Senior", "Python", "yes", "yes", "True"],
            ["Mid", "Python", "no", "yes", "True"],
            ["Mid", "Java", "yes", "no", "True"],
            ["Junior", "Python", "no", "yes", "False"]
    ]

    X, y = myutils.split_x_y_train(table)
    x_train, x_test, y_train, y_test = myevaluation.train_test_split(X, y, math.floor(len(table)*0.33), shuffle=True)
    remainder = []
    for i in range(len(x_train)):
        row = x_train[i]
        row.append(y_train[i])
        remainder.append(row)
    
    print(remainder)

    myRF = MyRandomForestClassifier()
    myRF.fit(remainder, 10, 100)

    y_predicted = myRF.predict(x_test)
    
    assert len(y_predicted) == len(y_test)

    count = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] == y_test[i]:
            count += 1
    
    assert count != 0
    