import pandas as pd
import random

def evaluate(model, data):
    # Usa validação cruzada e retorna as métricas pro model dado
    raise Exception("Not implemented")


def bootstrap(data, class_column, attr_sample_ratio):
    columns = set(data.columns.values)
    columns.remove(class_column)
    attr_count = len(columns)
    attr_samples_count = round(attr_count * attr_sample_ratio)
    columns_to_drop = random.sample(columns, attr_count - attr_samples_count)
    
    return data.sample(n=len(data), replace=True).drop(columns_to_drop, axis=1, inplace=False)


def f_measure():
    raise Exception("Not implemented")
