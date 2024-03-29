import pandas as pd
import numpy as np
import random

def evaluate(model, test_data, class_column, class_column_values):
    confusion_matrix = calculate_confusion_matrix(model, test_data, class_column, class_column_values)
    accuracy = calculate_accuracy(confusion_matrix, class_column_values)
    precision = calculate_precision(confusion_matrix, class_column_values)
    recall = calculate_recall(confusion_matrix, class_column_values)
    mean_precision = np.mean(list(precision.values()))
    mean_recall = np.mean(list(recall.values()))
    f1_score = f1_measure(mean_precision, mean_recall)

    #print(','.join(str(x) for x in [f1_score,accuracy,mean_precision,mean_recall]))

    print(32*'=')
    print('Accuracy: {:.4f}'.format(accuracy))
    for value in class_column_values:
        print('Class {} precision: {:.4f}'.format(value, precision[value]))
        print('Class {} recall: {:.4f}'.format(value, recall[value]))
    print('Mean precision: {:.4f}'.format(mean_precision))
    print('Mean recall: {:.4f}'.format(mean_recall))
    print('F1-score: {:.4f}'.format(f1_score))
    print(32*'=')

    return f1_score

def bootstrap(data):
    bootstrap = data.sample(n=len(data), replace=True)
    return bootstrap

def f1_measure(precision, recall):
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def k_cross_fold(data, k=10):
    folds = np.array_split(data, k)
    for i in range(k):
        train = folds.copy()
        train.pop(i)
        train = pd.concat(train, sort=False)
        test = folds[i]
        yield train, test

def stratified_k_cross_fold(data, class_column, k=10):
    data = data.sample(frac=1).reset_index(drop=True) # Shuffle DataFrame rows

    folds_per_class = []
    for _, g in data.groupby(class_column):
        folds_per_class.append(k_cross_fold(g, k=k))  # create fold for each class
    
    for _ in range(k):
        train = pd.DataFrame()
        test = pd.DataFrame()
        for c in range(len(folds_per_class)):         # append the next fold for each class
            train_c, test_c = next(folds_per_class[c])
            train = train.append(train_c)
            test = test.append(test_c)
        yield train, test

def stratified_k_cross_validation(model, data, class_column, k=10):
    class_column_values = data[class_column].unique()
    k_folds = stratified_k_cross_fold(data, class_column, k)
    scores = []
    #print('f1,acc,meanprec,meanrec')
    for _ in range(k):
        train, test = next(k_folds)
        model.train(train, class_column)
        f1_score = evaluate(model, test, class_column, class_column_values)
        scores.append(f1_score) 

    print('Average F1-score: {:.4f}'.format(np.mean(scores)))
    print('Standard deviation: {:.4f}'.format(np.std(scores)))
    print(32*'=')

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

def calculate_accuracy(confusion_matrix, class_column_values):
    acc = 0
    total = 0
    for value in class_column_values:
        acc += confusion_matrix[value][value]
        total += sum(confusion_matrix[value][i] for i in class_column_values)
    return acc / total

def calculate_precision(confusion_matrix, class_column_values):
    classes_precision = {}
    for value in class_column_values:
        try:
            vp = confusion_matrix[value][value]
            vp_fp = sum([confusion_matrix[i][value] for i in class_column_values])
            precision =  vp / vp_fp
        except ZeroDivisionError:
            precision = 0
        finally:
            classes_precision[value] = precision
    return classes_precision

def calculate_recall(confusion_matrix, class_column_values):
    classes_recall = {}
    for value in class_column_values:
        vp = confusion_matrix[value][value]
        vp_fn = sum([confusion_matrix[value][i] for i in class_column_values])
        recall =  vp / vp_fn
        classes_recall[value] = recall
    return classes_recall
