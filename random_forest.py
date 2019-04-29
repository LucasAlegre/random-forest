import argparse
import pandas as pd
import numpy as np
import random
from math import sqrt
from random_forest.random_forest import RandomForest
from random_forest.random_tree import RandomTree
from random_forest.util import bootstrap, evaluate


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="Random Forest - Aprendizado de Máquina 2019/1 UFRGS")
    prs.add_argument("-s", dest="seed", type=int, default=None, required=False, help="The random seed.\n")
    prs.add_argument("-d", dest="data", required=True, help="The dataset .csv file.\n")
    prs.add_argument("-c", dest="class_column", required=True, help="The column of the .csv to be predicted.\n")
    prs.add_argument("-sep", dest="sep", default=',', required=False, help=".csv separator.\n")
    prs.add_argument("-n", dest="num_trees", type=int, default=5, required=False, help="The number of trees in the random forest.\n")
    prs.add_argument("-v", action='store_true', default=False, required=False, help="View random tree image.\n")

    args = prs.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data, sep=args.sep)
    class_column = args.class_column

    attr_sample_size = int(sqrt(len(df.columns.values)))

    forest = RandomForest(args.num_trees, seed=args.seed)

    train = df.sample(frac=0.7, random_state=args.seed)
    test = df.loc[~df.index.isin(train.index)]

    forest.train(train, class_column, attr_sample_size)
    
    if args.v:
        forest.view_forest('RandomForest')

    evaluate(forest, test, class_column)
