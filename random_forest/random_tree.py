from math import log2


class RandomTree:

    AttributeMetadata = None

    def __init__(self):
        self.data = None
        self.class_column = None  # class to be predicted
        self.attributes = None
        self.root = None

    def train(self, data, class_column):
        self.data = data
        self.class_column = class_column
        self.attributes = set(self.data.columns.values)
        self.attributes.remove(self.class_column)  # all columns except the class
        RandomTree.AttributeMetadata = {atr: data[atr].unique().tolist() for atr in self.attributes}

        self.root = RandomTreeNode(data, self.attributes.copy(), self.class_column)
       
    def predict(self, instance):
        return self.root.predict(instance)

    def print_tree(self):
        self.root.print_node()


class RandomTreeNode:

    def __init__(self, data, attributes, class_column, is_leaf=False, terminal_class=None):
        self.is_leaf = is_leaf
        self.is_numerical = None
        self.class_column = class_column
        self.cut_point = None
        self.terminal_class = terminal_class
        self.children = dict()

        if self._number_of_classes(data) == 1:  # if all instances have the same class
            self.is_leaf = True
            self.terminal_class = data[class_column].values[0]
            return
        
        if len(attributes) == 0:  # if there is no more attributes to divide
            self.is_leaf = True
            self.terminal_class = data[class_column].value_counts().idxmax()
            return

        entropy = self.entropy(data)
        self.node_attribute = max(attributes, key=lambda a: entropy - self.entropy_attribute(data, a))  # attribute with the max gain (entropy - entropy of the attribute)
        self.is_numerical = data[self.node_attribute].dtype != object  # (object == str on Pandas)
        attributes.remove(self.node_attribute)

        for v in RandomTree.AttributeMetadata[self.node_attribute]:
            dv = data[data[self.node_attribute] == v]
            if len(dv) == 0:
                self.is_leaf = True
                self.terminal_class = data[class_column].value_counts().idxmax()   # most frequent class
                return
            else:
                self.children[v] = RandomTreeNode(dv, attributes.copy(), class_column)

    def predict(self, instance):
        if self.is_leaf:
            return self.terminal_class
        if self.is_numerical:
            return self.children[instance[self.node_attribute] < self.cut_point].predict(instance)
        
        return self.children[instance[self.node_attribute]].predict(instance)

    def _number_of_classes(self, data):
        return data[self.class_column].nunique()

    def entropy(self, data):
        n = len(data)
        value_counts = data[self.class_column].value_counts().tolist()
        return sum((-x/n)*log2(x/n) for x in value_counts)
    
    def entropy_attribute(self, data, attribute):
        n = len(data)
        mean_entropy = 0

        if data[attribute].dtype == object:  # categorical attribute (object == str on Pandas)
            for _, g in data.groupby(attribute):
                g_size = len(g)
                value_counts = g[self.class_column].value_counts().tolist()
                entropy = sum((-x/g_size)*log2(x/g_size) for x in value_counts)
                mean_entropy += (g_size/n)*entropy
        
        else:  # numerical attribute
            pass
        
        return mean_entropy
    
    def print_node(self, height=0):
        if self.is_leaf:
            print('{} {}'.format(height*'--', self.terminal_class))
        else:
            print('{} {}'.format(height*'--', self.node_attribute))
            for c in self.children.values():
                c.print_node(height + 1)

    
        