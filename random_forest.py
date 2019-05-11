import argparse
import pandas as pd
import numpy as np
import random
from math import sqrt
from random_forest.random_forest import RandomForest
from random_forest.random_tree import RandomTree
from random_forest.util import stratified_k_cross_validation


if __name__ == '__main__':

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="Random Forest - Aprendizado de Máquina 2019/1 UFRGS")

    prs.add_argument("-s",    dest="seed",         required=False, default=None,                help="The random seed.\n", type=int)
    prs.add_argument("-d",    dest="data",         required=False, default='datasets/wine.csv', help="The dataset .csv file.\n")
    prs.add_argument("-c",    dest="class_column", required=False, default='class',             help="The column of the .csv to be predicted.\n")
    prs.add_argument("-sep",  dest="sep",          required=False, default=',',                 help=".csv separator.\n")
    prs.add_argument("-n",    dest="num_trees",    required=False, default=5,                   help="The number of trees in the random forest.\n", type=int)
    prs.add_argument("-k",    dest="num_folds",    required=False, default=10,                  help="The number of folds used on cross validation.\n", type=int)
    prs.add_argument('-drop', nargs='+',           required=False, default=[],                  help="Columns to drop from .csv.")
    
    prs.add_argument("-not-sample",  action='store_true', required=False, default=False, help="Do not sample attributes on each node.\n")
    prs.add_argument("-cut-by-mean", action='store_true', required=False, default=False, help="Cut point by mean of numerical attribute.\n")
    prs.add_argument("-v",           action='store_true', required=False, default=False, help="View random tree image.\n")

    args = prs.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    
    df = pd.read_csv(args.data, sep=args.sep)
    class_column = args.class_column

    for column in args.drop:
        df.drop(column, inplace=True, axis=1)

    if args.not_sample:
        attr_sample_size = None
    else:
        attr_sample_size = int(sqrt(len(df.columns.values)))

    forest = RandomForest(args.num_trees, attr_sample_size=attr_sample_size, cut_point_by_mean=args.cut_by_mean)

    stratified_k_cross_validation(forest, df, class_column, k=args.num_folds)
    
    if args.v:
        forest.view_forest('RandomForest')
