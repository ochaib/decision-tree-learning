import math
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

    def add_left_child(self, child):
        self.left = child

    def add_right_child(self, child):
        self.right = child

    @property
    def is_leaf(self):
        return (self.left is None) & (self.right is None)


# Not necessary thanks to numpy
# def same_labels(dataset):
#     # Labels are located at the 7th index of the datasets.
#     tdBools = [dataset[i][7] == dataset[i - 1][7] for i in range(len(dataset))]
#     return all(tdBools)


def function_h(np_dataset):
    # Sum of pk where pk is the number of samples with label k divided
    # by total number of samples from initial dataset, for each label
    # from 1 to k multiplied by the log2 of pk. Negated.

    # Extract label column from dataset.
    labels = np_dataset[:, 6]
    n_labels = len(labels)
    psum = 0

    for i in np.unique(labels):
        i_elements = labels[labels == i]
        p = len(i_elements)/n_labels
        psum += p * math.log(p, 2)

    return -psum


def remainder(l_dataset, r_dataset):
    n_samples_left = np.shape(l_dataset)[0]
    n_samples_right = np.shape(r_dataset)[0]
    l_remainder = (n_samples_left/(n_samples_left + n_samples_right)) * function_h(l_dataset)
    r_remainder = (n_samples_right/(n_samples_left + n_samples_right)) * function_h(r_dataset)
    return l_remainder + r_remainder


def evaluate_information_gain(np_dataset, l_dataset, r_dataset):
    return function_h(np_dataset) - remainder(l_dataset, r_dataset)


# This may be an old solution and a modern one may be out.
def split_on_cond(array, cond):
    return [array[cond], array[~cond]]


def find_split(dataset):
    # First find good split points by sorting the values of the attribute
    # (there are seven attributes)
    # So maybe for each attribute, find a point (value) that is between two
    # examples in sorted order i.e. only one value exists for that attribute
    # While keeping track of the running totals? of positive and negative
    # examples on each side of the split point??

    # Highest information gain.
    highest_information_gain = 0
    # Highest information gain attribute, value, left, right and sorted datasets.
    hig_attribute = None
    hig_value = None
    hig_sorted_dataset = None
    hig_l_dataset = None
    hig_r_dataset = None

    # Iterate through attributes e.g. 0 to 6
    for i in range(np.shape(dataset)[1] - 2):
        # Return dataset sorted by ith attribute (column) value
        dataset = dataset[dataset[:, i].argsort()]
        # Look for independent values also recording what is before and after them.
        # Should return independent values of attribute sorted correctly.
        independent_values = np.unique(dataset[:, i], axis=0)

        # Now to split dataset on independent values to retrieve sets on either side of split.
        for v in independent_values:
            # Dataset split on value, split_dataset[0] is <= value and split_dataset[1] is > value.
            # split_dataset = np.split(dataset, np.where(dataset[:, i] > v))
            split_dataset = split_on_cond(dataset, dataset[:, i] > v)

            # Calculate the information gain for each value for this attribute.
            current_ig = evaluate_information_gain(dataset, split_dataset[0], split_dataset[1])

            if current_ig > highest_information_gain:
                highest_information_gain = current_ig
                hig_attribute = i
                hig_value = v
                hig_sorted_dataset = dataset
                hig_l_dataset = split_dataset[0]
                hig_r_dataset = split_dataset[1]

    return hig_attribute, hig_value, hig_sorted_dataset, hig_l_dataset, hig_r_dataset


def decision_tree_learning(training_dataset, depth):
    # Load training dataset file as numpy array.
    np_dataset = np.loadtxt(training_dataset)

    if len(np.unique(np_dataset[:, 7], axis=0)) == 1:
        # Attribute refers to the index or a column of the matrix, training_dataset.
        # Create a new leaf TreeNode with the label (which is they same for all
        # entries in the dataset) as the value.
        # But looking at the dataset just because the label is the same doesn't
        # mean that the values are the same so which value are we choosing?
        return TreeNode(np_dataset[0][7]), depth
    else:
        (attr, value, dataset, l_dataset, r_dataset) = find_split(np_dataset)
        # Return a new decision tree with root as value,
        # i.e. left and right child nodes are yet to be created.
        node = TreeNode(attr, value, None, None)
        (l_branch, l_depth) = decision_tree_learning(l_dataset, depth + 1)
        (r_branch, r_depth) = decision_tree_learning(r_dataset, depth + 1)
        return node, max(l_depth, r_depth)


# Takes a trained tree and a test dataset and returns the accuracy of the tree.
# Use 10-fold cross validation on both clean and noisy datasets to evaluate
# decision tree.
def evaluate(test_db, trained_tree):
    pass