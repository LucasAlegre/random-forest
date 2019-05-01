import pandas as pd
import numpy as np
import random

def evaluate(model, test_data, class_column, class_column_values):
    confusion_matrix = calculate_confusion_matrix(model, test_data, class_column, class_column_values)
    return f1_measure(confusion_matrix, class_column_values)

def bootstrap(data):
    bootstrap = data.sample(n=len(data), replace=True)
    return bootstrap

def f1_measure(confusion_matrix, class_column_values):
    precision = calculate_precision(confusion_matrix, class_column_values)
    recall    = calculate_recall(confusion_matrix, class_column_values)
    f1 = (2 * precision * recall) / (precision + recall)

    print(f1)

    return f1

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

def calculate_confusion_matrix(model, test_data, class_column, class_column_values):
    confusion_matrix = {}

    # Initialize confusion matrix with zeros
    for true_class in class_column_values:
        confusion_matrix[true_class] = {}
        for predicted_class in class_column_values:
            confusion_matrix[true_class][predicted_class] = 0

    for index, row in test_data.iterrows():
        true_class = row[class_column]
        predicted_class = model.predict(row)

        confusion_matrix[true_class][predicted_class] += 1

    return confusion_matrix

def calculate_precision(confusion_matrix, class_column_values):
    classes_precision = []
    for value in class_column_values:
        vp = confusion_matrix[value][value]
        vp_fp = sum([confusion_matrix[i][value] for i in class_column_values])
        precision =  vp / vp_fp
        classes_precision.append(precision)

    return sum(classes_precision) / len(classes_precision)

def calculate_recall(confusion_matrix, class_column_values):
    classes_recall = []
    for value in class_column_values:
        vp = confusion_matrix[value][value]
        vp_fp = sum([confusion_matrix[value][i] for i in class_column_values])
        recall =  vp / vp_fp
        classes_recall.append(recall)

    return sum(classes_recall) / len(classes_recall)
