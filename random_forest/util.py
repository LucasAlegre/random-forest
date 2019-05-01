import pandas as pd
import random

def evaluate(model, test_data, class_column):
    c = 0
    for index, row in test_data.iterrows():
        if model.predict(row) == row[class_column]:
            c += 1
    
    print(c/len(test_data))


def bootstrap(data, seed):
    bootstrap = data.sample(n=len(data), replace=True, random_state=seed)
    return bootstrap


def f_measure():
    raise Exception("Not implemented")
