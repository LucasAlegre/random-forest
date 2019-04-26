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

    tree = RandomTree()
    train = df.sample(frac=0.7)
    test = df.loc[~df.index.isin(train.index)]
    tree.train(train, class_column)
   
    tree.view_tree('DecisionTree')

    c = 0
    for index, row in test.iterrows():
        if tree.predict(row) == row[class_column]:
            c += 1
    print(c/len(test))

