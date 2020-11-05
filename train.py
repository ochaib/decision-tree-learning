from constants import *
import math
import numpy as np
from tree import TreeNode


# utility functions for training
def function_h(np_dataset):
    # Sum of pk where pk is the number of samples with label k divided
    # by total number of samples from initial dataset, for each label
    # from 1 to k multiplied by the log2 of pk. Negated.

    # Extract label column from dataset.
    labels = np_dataset[:, -1]
    n_labels = len(labels)
    psum = 0

    for i in np.unique(labels):
        i_elements = labels[labels == i]
        p = len(i_elements) / n_labels
        psum += p * math.log(p, 2)

    return -psum


def remainder(l_dataset, r_dataset):
    n_samples_left = np.shape(l_dataset)[0]
    n_samples_right = np.shape(r_dataset)[0]
    l_remainder = (n_samples_left / (n_samples_left + n_samples_right)) * function_h(l_dataset)
    r_remainder = (n_samples_right / (n_samples_left + n_samples_right)) * function_h(r_dataset)
    return l_remainder + r_remainder


def evaluate_information_gain(np_dataset, l_dataset, r_dataset):
    return function_h(np_dataset) - remainder(l_dataset, r_dataset)


# training functions for splitting tree
def split_on_cond(array, cond):
    return [array[cond], array[~cond]]


def find_split(dataset):
    # First find good split points by sorting the values of the attribute
    # (there are seven attributes)

    # Since we have ordered (real) values:
    # For each feature, sort its values, and consider only split points that
    # are between two examples with different class labels.

    # While keeping track of the running totals? of positive and negative
    # examples on each side of the split point??

    # Highest information gain.
    highest_information_gain = None
    # Highest information gain attribute, value, left, right and sorted datasets.
    hig_attribute = None
    hig_value = None
    hig_sorted_dataset = None
    m, n = dataset.shape
    hig_l_dataset = np.empty([m, n])
    hig_r_dataset = np.empty([m, n])

    # Iterate through attributes e.g. 0 to 6
    for i in range(np.shape(dataset)[1] - 1):
        # Return dataset sorted by ith attribute (column) value
        dataset = dataset[dataset[:, i].argsort()]
        # Look for independent values also recording what is before and after them.
        # Should return independent values of attribute sorted correctly.

        # Now to split dataset on isolated values, ones that are between two examples
        # with different class labels (last column), to retrieve sets on either side of split.
        for j in range(len(dataset[:, i]) - 1):
            # Check if value v is isolated the values above and below it must belong to
            # different labels (column LABEL_INDEX), continue if true, skip if not

            # Edge case isolation check
            if len(dataset[:, i]) > 1:
                # Main value isolation check
                if dataset[j, LABEL_INDEX] == dataset[j + 1, LABEL_INDEX]:
                    continue

            split_candidate = dataset[j, i]
            # Dataset split on value, split_dataset[0] is <= value and split_dataset[1] is > value.
            split_dataset = split_on_cond(dataset, dataset[:, i] <= split_candidate)

            # Calculate the information gain for each value for this attribute.
            current_ig = evaluate_information_gain(dataset, split_dataset[0], split_dataset[1])

            if not highest_information_gain or current_ig > highest_information_gain:
                highest_information_gain = current_ig
                hig_attribute = i
                hig_value = split_candidate
                hig_sorted_dataset = dataset
                hig_l_dataset = split_dataset[0]
                hig_r_dataset = split_dataset[1]

    return hig_attribute, hig_value, hig_sorted_dataset, hig_l_dataset, hig_r_dataset


# takes recursive steps to build a tree from given dataset
def train(training_dataset, depth=1):
    if len(np.unique(training_dataset[:, LABEL_INDEX])) == 1:
        # Attribute refers to the index or a column of the matrix, training_dataset.
        # Create a new leaf TreeNode with the label (which is they same for all
        # entries in the dataset) as the value.
        # But looking at the dataset just because the label is the same doesn't
        # mean that the values are the same so which value are we choosing?
        node = TreeNode(training_dataset[0][LABEL_INDEX])
        node.count = len(training_dataset)
        return node, depth
    else:
        (attr, value, dataset, l_dataset, r_dataset) = find_split(training_dataset)
        # Return a new decision tree with root as value,
        # i.e. left and right child nodes are yet to be created.
        node = TreeNode(value, attr, None, None)
        (l_branch, l_depth) = train(l_dataset, depth + 1)
        (r_branch, r_depth) = train(r_dataset, depth + 1)
        node.add_left_child(l_branch)
        node.add_right_child(r_branch)
        return node, max(l_depth, r_depth)



        

        