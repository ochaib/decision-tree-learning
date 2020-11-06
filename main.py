import sys
import numpy as np
from train import train, prune_tree
from evaluate import evaluate, calculate_measures
from visualizer import visualize


def generate_test_training(dataset, k):
    """
    Split the given dataset to produce k different collections of
    training/test datasets with the test dataset taking a different fold
    each time and the training dataset being a concatenation of the remaining
    datasets.
    :param dataset: Dataset to be split.
    :param k: Fold count
    :type dataset: np.array
    :type k: int
    :return: Training datasets, test datasets
    :rtype: np.array, np.array
    """
    # Shuffle the test dataset
    np.random.shuffle(dataset)

    # Divide the dataset into k equal folds/splits.
    folds = np.array_split(dataset, k)

    # Use k-1 folds for training+validation and 1 for testing.
    training_sets = []
    test_sets = []
    for i in range(k):
        tmp = folds.copy()
        test_sets.append(tmp.pop(i))
        training_sets.append(tmp)
    # Concatenate numpy arrays
    return np.concatenate(np.asarray(training_sets)), np.asarray(test_sets)


# Takes a trained tree and a test dataset and returns the accuracy of the tree.
# Use 10-fold cross validation on both clean and noisy datasets to evaluate
# decision tree.
def main(dataset):
    np_dataset = np.loadtxt(dataset)
    k = 10
    accuracies = []
    pruned_accuracies = []
    training_sets, test_sets = generate_test_training(np_dataset, k)
    agg_confusion_matrix = np.zeros((4, 4))
    agg_pruned_confusion_matrix = np.zeros((4, 4))

    # Evaluation on unpruned tree
    for i in range(k):
        training_db = training_sets[i]
        test_db = test_sets[i]
        # Train
        trained_tree, depth = train(training_db, 1)
        # Evaluate
        (accuracy, confusion_matrix) = evaluate(test_db, trained_tree)
        agg_confusion_matrix += confusion_matrix
        accuracies.append(accuracy)
    # Calculate average accuracy
    agg_confusion_matrix /= k
    print(agg_confusion_matrix)
    calculate_measures(agg_confusion_matrix)
    average_accuracy = np.average(accuracies)

    # Tree pruning
    inner_training_sets, validation_sets = generate_test_training(training_sets, k - 1)
    # Evaluation on pruned tree
    for i in range(k):
        test_db = test_sets[i]
        for j in range(k - 1):
            training_db = inner_training_sets[j, i]
            validation_db = validation_sets[j, i]
            # Train
            trained_tree, depth = train(training_db, 1)
            # Evaluation
            (accuracy, confusion_matrix) = evaluate(validation_db, trained_tree)
            # Prune
            pruned_tree = prune_tree(trained_tree, validation_db, accuracy)
            # Evaluate on now pruned tree
            (pruned_accuracy, pruned_confusion_matrix) = evaluate(test_db, pruned_tree)
            pruned_accuracies.append(pruned_accuracy)
            agg_pruned_confusion_matrix += pruned_confusion_matrix
    agg_pruned_confusion_matrix /= (k * k - 1)
    print(agg_pruned_confusion_matrix)
    calculate_measures(agg_pruned_confusion_matrix)
    avg_pruned_accuracy = np.average(pruned_accuracies)

    print("Average Accuracy: ", average_accuracy)
    print("Average Accuracy of Pruned Decision Tree: ", avg_pruned_accuracy)

    # Train on the entire dataset
    tree, depth = train(np_dataset)
    # Visualize this, saving it to an aptly named file
    visualize(tree, depth, dataset[:dataset.rfind('.')] + '.png')


if __name__ == "__main__":
    main(sys.argv[-1])
