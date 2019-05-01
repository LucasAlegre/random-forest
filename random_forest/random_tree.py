from math import log2
import random
from graphviz import Digraph
import numpy as np
import pandas as pd


class RandomTree:

    AttributesDomain = dict()
    NumericalAttributes = set()
    Classes = set()

    def __init__(self):
        self.data = None
        self.class_column = None  # class to be predicted
        self.attributes = None
        self.root = None

    def train(self, data, class_column, attr_sample_size=None, cut_point_by_mean=False):
        self.data = data
        self.class_column = class_column

        self.attributes = self.data.columns.values.tolist()
        self.attributes.remove(self.class_column)  # all columns except the class

        self._compute_metadata()

        self.root = RandomTreeNode(data, self.attributes.copy(), self.class_column, attr_sample_size, cut_point_by_mean)
       
    def predict(self, instance):
        return self.root.predict(instance)

    def _compute_metadata(self):
        RandomTree.Classes.update(self.data[self.class_column].unique().tolist())
        for atr in self.attributes:
            if self.data[atr].dtype == object:  # Categorical Attribute (object == str on Pandas)
                RandomTree.AttributesDomain[atr] = self.data[atr].unique().tolist()
            else:
                RandomTree.NumericalAttributes.add(atr)

    def print_tree(self):
        self.root.print_node()

    def view_tree(self, name):
        g = Digraph(name)
        g.node_attr.update(style='filled', fontname='Arial')
        g.edge_attr.update(fontname='Arial')
        nodes = [self.root]
        while len(nodes) != 0:
            node = nodes.pop(0)
            g.node(str(id(node)), str(node), shape="box" if node.is_leaf else "ellipse", color="gray" if node.is_leaf else "green3")
            for value, child in node.children.items():
                g.edge(str(id(node)), str(id(child)), label=str(value))
                nodes.append(child)
        g.view()


class RandomTreeNode:

    def __init__(self, data, attributes, class_column, attr_sample_size=None, cut_point_by_mean=False):
        self.is_leaf = False
        self.class_column = class_column
        self.cut_point_by_mean = cut_point_by_mean
        self.cut_point = None
        self.terminal_class = None
        self.children = dict()
        self.num_samples = len(data)
        self.data_entropy = self.entropy(data)

        if self.num_samples == 0:
            self.is_leaf = True
            self.terminal_class = random.sample(RandomTree.Classes, 1)[0]
            return

        if self._number_of_classes(data) == 1:  # if all instances have the same class
            self.is_leaf = True
            self.terminal_class = data[class_column].values[0]
            return
        
        if len(attributes) == 0:  # if there is no more attributes to divide
            self.is_leaf = True
            self.terminal_class = data[class_column].value_counts().idxmax()
            return

        if attr_sample_size != None:
            sample_attributes = random.sample(attributes, min(attr_sample_size, len(attributes)))
        else:
            sample_attributes = attributes

        attr_entropies = {attr: self.entropy_attribute(data, attr) for attr in sample_attributes}
        self.node_attribute = max(sample_attributes, key=lambda attr: self.data_entropy - attr_entropies[attr]['entropy'])  # attribute with the max gain (entropy - entropy of the attribute)
        self.gain = self.data_entropy - attr_entropies[self.node_attribute]['entropy']

        #if self.node_attribute not in RandomTree.NumericalAttributes or self.cut_point_by_mean:
        attributes.remove(self.node_attribute)

        # Categorial Attribute
        if self.node_attribute not in RandomTree.NumericalAttributes:
            for v in RandomTree.AttributesDomain[self.node_attribute]:
                dv = data[data[self.node_attribute] == v]
                if len(dv) == 0:
                    self.is_leaf = True
                    self.terminal_class = data[class_column].value_counts().idxmax()   # most frequent class
                    self.children = {}
                    return
                else:
                    self.children[v] = RandomTreeNode(dv, attributes.copy(), class_column, attr_sample_size, cut_point_by_mean)
        # Numerical Attribute
        else:
            self.cut_point = attr_entropies[self.node_attribute]['cut-point']
            subset_greater_than_cut = data[data[self.node_attribute] > self.cut_point]
            less_or_equal_cut = data[data[self.node_attribute] <= self.cut_point]
            self.children = {
                False: RandomTreeNode(subset_greater_than_cut, attributes.copy(), class_column, attr_sample_size, cut_point_by_mean),
                True: RandomTreeNode(less_or_equal_cut, attributes.copy(), class_column, attr_sample_size, cut_point_by_mean)
            }

    def predict(self, instance):
        if self.is_leaf:
            return self.terminal_class
        if self.node_attribute in RandomTree.NumericalAttributes:
            return self.children[instance[self.node_attribute] <= self.cut_point].predict(instance)
        
        return self.children[instance[self.node_attribute]].predict(instance)

    def _calculate_cut_points(self, data, attribute):
        if self.cut_point_by_mean:
            return [data[attribute].mean()]
        else:
            cut_points = []
            sorted_data = data.sort_values(attribute)
            for (_, row1),(_, row2) in zip(sorted_data[:-1].iterrows(), sorted_data[1:].iterrows()):
                if row1[self.class_column] != row2[self.class_column]:
                    cut_points.append((row1[attribute] + row2[attribute]) / 2)
            return list(set(cut_points))

    def _number_of_classes(self, data):
        return data[self.class_column].nunique()

    def entropy(self, data):
        n = len(data)
        value_counts = data[self.class_column].value_counts().tolist()
        return sum((-x/n)*log2(x/n) for x in value_counts)
    
    def entropy_attribute(self, data, attribute):
        # categorical attribute
        if attribute not in RandomTree.NumericalAttributes:  
            return {'entropy': self.entropy_categorical_attribute(data, attribute)}
        # numerical attribute
        else:  
            cut_points = self._calculate_cut_points(data, attribute)
            entropies = [self.entropy_numerical_attribute(data, attribute, cut_point) for cut_point in cut_points]
            max_entropy = max(entropies)
            cut_point = cut_points[entropies.index(max_entropy)]
            return {'entropy': max_entropy, 'cut-point': cut_point}
    
    def entropy_categorical_attribute(self, data, attribute):
        n = len(data)
        mean_entropy = 0

        for _, g in data.groupby(attribute):
            g_size = len(g)
            value_counts = g[self.class_column].value_counts().tolist()
            entropy = sum((-x/g_size)*log2(x/g_size) for x in value_counts)
            mean_entropy += (g_size/n)*entropy

        return mean_entropy

    def entropy_numerical_attribute(self, data, attribute, cut_point):
        n = len(data)
        mean_entropy = 0       
        subset_greater_than_cut = data[data[attribute] > cut_point]
        less_or_equal_cut = data[data[attribute] <= cut_point]

        for g in [subset_greater_than_cut, less_or_equal_cut]:
            g_size = len(g)
            value_counts = g[self.class_column].value_counts().tolist()
            entropy = sum((-x/g_size)*log2(x/g_size) for x in value_counts)
            mean_entropy += (g_size/n)*entropy

        return mean_entropy
    
    def print_node(self, height=0):
        if self.is_leaf:
            print('{} {}'.format(height*'--', self.terminal_class))
        else:
            print('{} {}'.format(height*'--', self.node_attribute))
            for c in self.children.values():
                c.print_node(height + 1)
    
    def __str__(self):
        if self.is_leaf:
            return "{}\n\nEntropy: {:.3f}\nSamples: {}".format(self.terminal_class, self.data_entropy, self.num_samples)
        else:
            if self.node_attribute in RandomTree.NumericalAttributes:
                attribute = "{} <= {:.2f}".format(self.node_attribute, self.cut_point)
            else:
                attribute = self.node_attribute
            return "{}\n\nEntropy: {:.3f}\nGain: {:.3f}\nSamples: {}".format(attribute, self.data_entropy, self.gain, self.num_samples)
 