import pandas as pd
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
