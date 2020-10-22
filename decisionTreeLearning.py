import numpy as np

class TreeNode:
    treeNode = {}

    def __init__(self, value, attr=None, left=None, right=None):
        self.value = value
        self.attr = attr
        self.left = left
        self.right = right

    def TreeNode(self):
        return {self.attr, self.value, self.left, self.right}

    def add_leftChild(self, child):
        self.left = child

    def add_rightChild(self, child):
        self.right = right

    def isLeaf(self):
        return (self.left is None) && (self.right is None)


def decision_tree_learning(training_dataset, depth):
    if sameLabels(training_dataset) then:
        # Attribute refers to the index or a column of the matrix, training_dataset.
        # Create a new leaf TreeNode with the label (which is they same for all
        # entries in the dataset) as the value.
        # But looking at the dataset just because the label is the same doesn't
        # mean that the values are the same so which value are we choosing?
        return (TreeNode(training_dataset[0][7]), depth)
    else:
        split = findSplit(training_dataset)
        (l_dataset, r_dataset) = splitOn(split)
        node = TreeNode(split)
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth+1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth+1)
        return (node, max(l_depth, r_depth))

def sameLabels(training_dataset):
    # Labels are located at the 7th index of the datasets.
    tdBools = [training_dataset[i][7] == training_dataset[i-1][7] for i in range(len(training_dataset))]
    return all(tdBools)

def findSplit(training_dataset):
