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
        self.right = child

    @property
    def isLeaf(self):
        return (self.left is None) & (self.right is None)


def decision_tree_learning(training_dataset, depth):
    # Load training dataset file as numpy array.
    np_dataset = np.loadtxt(training_dataset)

    if len(np.unique(np_dataset[:, 6], axis=0)) == 1:
        # Attribute refers to the index or a column of the matrix, training_dataset.
        # Create a new leaf TreeNode with the label (which is they same for all
        # entries in the dataset) as the value.
        # But looking at the dataset just because the label is the same doesn't
        # mean that the values are the same so which value are we choosing?
        return TreeNode(np_dataset[0][7]), depth
    else:
        split = findSplit(np_dataset)
        (l_dataset, r_dataset) = splitOn(split)
        node = TreeNode(split)
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return node, max(l_depth, r_depth)


# Not necessary thanks to numpy
def sameLabels(dataset):
    # Labels are located at the 7th index of the datasets.
    tdBools = [dataset[i][7] == dataset[i - 1][7] for i in range(len(dataset))]
    return all(tdBools)


def findSplit(dataset):
    pass


def splitOn(split):
    pass


def evaluateInformationGain(np_dataset, l_dataset, r_dataset):
    return functionH(np_dataset) - remainder(l_dataset, r_dataset)


def functionH(np_dataset):
    # Sum of pk where pk is the number of samples with label k divided
    # by total number of samples from initial dataset, for each label
    # from 1 to k multiplied by the log2 of pk. Negated.

    # Extract label column from dataset.
    labels = np_dataset[:, 6]

    np.log2()


def remainder(l_dataset, r_dataset):
    pass
