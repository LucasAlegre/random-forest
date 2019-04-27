import pandas as pd
import numpy as np
from random_forest.random_forest import RandomForest
from random_forest.random_tree import RandomTree
from random_forest.util import bootstrap, evaluate


if __name__ == '__main__':

    #np.random.seed(42)

    #df = pd.read_csv('datasets/dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    #class_column = 'Joga'
    
    #df = pd.read_csv('datasets/ionosphere.csv')
    #class_column = 'g/b'

    #df = pd.read_csv('datasets/wine.csv')
    #class_column = 'class'

    df = pd.read_csv('datasets/wdbc.csv')
    df.drop('id', axis=1, inplace=True)
    class_column = 'diagnosis'

    #df = pd.read_csv('datasets/weatherAUS.csv')
    #class_column = 'RainTomorrow'
    #df.drop('RISK_MM', axis=1, inplace=True)

    #df = pd.read_csv('datasets/pulsar_stars.csv')
    #class_column = 'target_class'

    num_trees = 5
    attr_sample_ratio = 1

    forest = RandomForest(num_trees)
    train = df.sample(frac=0.7)
    test = df.loc[~df.index.isin(train.index)]
    forest.train(train, class_column, attr_sample_ratio)
   
    forest.view_forest('RandomForest')

    c = 0
    for index, row in test.iterrows():
        if forest.predict(row) == row[class_column]:
            c += 1
    print(c/len(test))
