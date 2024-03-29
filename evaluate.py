from constants import *
import numpy as np


def predict_value(features, trained_tree):
    node = trained_tree
    while not node.is_leaf:
        if features[node.attr] <= node.value:
            node = node.left
        else:
            node = node.right
    return node.value


def evaluate(test_db, trained_tree):
    """
    Evaluate trained tree against the dedicated given test set,
    producing a confusion matrix by counting TP, FP, TN, FN counts
    generated by predicting the value of trained tree using test_db and
    comparing to test_db label to classify.
    :param test_db: The dataset used to evaluate the tree.
    :param trained_tree: The trained decision tree to evaluate.
    :type test_db: np.array
    :type trained_tree: TreeNode
    :returns: Accuracy calculated from confusion matrix.
              Confusion matrix
    :rtype: float, np.array
    """
    # Retrieve number of labels for use in creation of confusion matrix.
    cm_shape = len(np.unique(test_db[:, LABEL_INDEX]))
    confusion_matrix = np.zeros((cm_shape, cm_shape))
    for i in range(len(test_db)):
        prediction = int(predict_value(test_db[i, :LABEL_INDEX], trained_tree))
        confusion_matrix[prediction - 1, int(test_db[i, LABEL_INDEX] - 1)] += 1
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return accuracy, confusion_matrix


def calculate_measures(confusion_matrix):
    """
    Using the averaged confusion matrix calculate the recall and precision
    rates per class, the F1-measures derived from the recall and precision
    rates of the previous step.
    :param confusion_matrix: Confusion matrix calculated from the
                             evaluation function.
    :return: Print the class and the associated measures, recall,
             precision, F1.
    """
    column_totals = np.sum(confusion_matrix, axis=0)
    row_totals = np.sum(confusion_matrix, axis=1)
    for i in range(len(confusion_matrix)):
        true_positives = confusion_matrix[i, i]
        precision = true_positives / (row_totals[i])
        recall = true_positives / (column_totals[i])
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'Class {i + 1}:\n'
              f'    Recall = {recall}\n'
              f'    Precision = {precision}\n'
              f'    F1 = {f1}')
