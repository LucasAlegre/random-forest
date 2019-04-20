import pandas as pd
from random_forest.random_forest import RandomForest
from random_forest.random_tree import RandomTree
from random_forest.util import bootstrap, evaluate


if __name__ == '__main__':

    df = pd.read_csv('datasets/dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    class_column = 'Joga'
    
    #df = pd.read_csv('datasets/ionosphere.csv')
    #class_column = 'g/b'

    #df = pd.read_csv('datasets/wine.csv')
    #class_column = 'class'

    #df = pd.read_csv('datasets/wdbc.csv')
    #df.drop('id', axis=1, inplace=True)
    #class_column = 'diagnosis'

    #df = pd.read_csv('datasets/weatherAUS.csv')
    #class_column = 'RainTomorrow'
    #df.drop('RISK_MM', axis=1, inplace=True)

    #df = pd.read_csv('datasets/pulsar_stars.csv')
    #class_column = 'target_class'

    tree = RandomTree()
    tree.train(df, class_column)
    tree.view_tree('DecisionTree')

