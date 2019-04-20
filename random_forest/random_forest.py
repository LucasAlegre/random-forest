from .random_tree import RandomTree
from .util import bootstrap


class RandomForest:

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = [RandomTree() for _ in range(self.num_trees)]
        raise Exception("Not implemented")
    
    def train(self, data, class_column):
        for tree in self.trees:
            tree.train(bootstrap(data), class_column)

    def predict(self, instance):
        # Majority Voting
        raise Exception("Not implemented")