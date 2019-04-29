from .random_tree import RandomTree
from .util import bootstrap
from graphviz import Digraph


class RandomForest:

    def __init__(self, num_trees, seed=None):
        self.seed = seed
        self.num_trees = num_trees
        self.trees = [RandomTree() for _ in range(self.num_trees)]
    
    def train(self, data, class_column, attr_sample_size=None):
        if self.num_trees > 1:
            bootstrap_size = int(len(data) * 0.8)
            for tree in self.trees:
                tree.train(bootstrap(data, size=bootstrap_size, seed=self.seed), class_column, attr_sample_size)
        else:
            self.trees[0].train(data, class_column, attr_sample_size=attr_sample_size)

    def predict(self, instance):
        # Majority Voting
        trees_predictions = [tree.predict(instance) for tree in self.trees]
        forest_prediction = max(set(trees_predictions), key=trees_predictions.count)
        return forest_prediction

    def view_forest(self, name):
        g = Digraph(name)
        g.graph_attr.update(fontsize='40')
        g.node_attr.update(style='filled', fontname='Arial')
        g.edge_attr.update(fontname='Arial')

        i = 0
        for tree in self.trees:
            i += 1
            with g.subgraph(name='cluster_' + str(i)) as c:
                nodes = [tree.root]
                while len(nodes) != 0:
                    node = nodes.pop(0)
                    c.node(str(id(node)), str(node), shape="box" if node.is_leaf else "ellipse", color="gray" if node.is_leaf else "green3")
                    for value, child in node.children.items():
                        c.edge(str(id(node)), str(id(child)), label=str(value))
                        nodes.append(child)
                c.attr(label='Tree ' + str(i))
        g.view()