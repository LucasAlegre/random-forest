import pandas as pd
import numpy as np
import random

def evaluate(model, test_data, class_column):
    c = 0
    for index, row in test_data.iterrows():
        if model.predict(row) == row[class_column]:
            c += 1
    
    print(c/len(test_data))

def bootstrap(data):
    bootstrap = data.sample(n=len(data), replace=True)
    return bootstrap

def f_measure():
    raise Exception("Not implemented")

def create_cross_validation_forests(df, num_trees, num_folds):
    from .random_forest import RandomForest
    
    forests = []
    folds = np.array_split(df, num_folds)

    for i in range(num_folds):
        train = folds.copy()
        train.pop(i)
        train = pd.concat(train, sort=False)
        test = folds[i]

        forest = RandomForest(num_trees, train, test)
        forests.append(forest)

    return forests
