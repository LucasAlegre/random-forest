import pandas as pd
from random_forest.random_forest import RandomForest
from random_forest.random_tree import RandomTree


if __name__ == '__main__':

    df = pd.read_csv('datasets/dadosBenchmark_validacaoAlgoritmoAD.csv', sep=';')
    class_column = 'Joga'
    
    #df = pd.read_csv('datasets/ionosphere.csv')
    #class_column = 'g/b'
    #print(df)

    #df = pd.read_csv('datasets/wine.csv')
    #class_column = 'class'
    #print(df)

    #df = pd.read_csv('datasets/wdbc.csv')
    #class_column = 'diagnosis'
    #print(df)

    tree = RandomTree()
    tree.train(df, class_column)
    tree.print_tree()

