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
    confusion_matrix = np.zeros((4, 4))
    for i in range(len(test_db)):
        prediction = int(predict_value(test_db[i, :LABEL_INDEX], trained_tree))
        confusion_matrix[prediction - 1, int(test_db[i, LABEL_INDEX] - 1)] += 1
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return accuracy, confusion_matrix


def calculate_measures(confusion_matrix):
    column_totals = np.sum(confusion_matrix, axis=0)
    row_totals = np.sum(confusion_matrix, axis=1)
    for i in range(len(confusion_matrix)):
        true_positives = confusion_matrix[i][i]
        false_positives = column_totals[i] \
                          - true_positives
        false_negatives = row_totals[i] \
                          - true_positives
        recall = true_positives / (true_positives + false_negatives)
        precision = true_positives / (true_positives + false_positives)
        f1 = (2 * precision * recall) / (precision + recall)
        print(f'''Class {i + 1}: recall = {recall}, 
                precision = {precision}, f1 = {f1}''')
