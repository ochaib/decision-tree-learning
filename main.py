import sys
import numpy as np
from train import train
from evaluate import evaluate, calculate_measures, prune_tree
from visualizer import visualize

def generate_test_training(dataset, k):
    # Shuffle test dataset
    np.random.shuffle(dataset)

    # Divide the dataset into k equal folds/splits.
    folds = np.array_split(dataset, k)

    # Use k-1 folds for training+validation and 1 for testing
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

    # Evaluation on unpruned tree.
    for i in range(k):
        training_db = training_sets[i]
        test_db = test_sets[i]
        # train
        trained_tree, depth = train(training_db, 1)
        # evaluate
        (accuracy, confusion_matrix) = evaluate(test_db, trained_tree)
        agg_confusion_matrix += confusion_matrix
        accuracies.append(accuracy)
    # Calculate average accuracy
    agg_confusion_matrix /= k
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
            # train
            trained_tree, depth = train(training_db, 1)
            # evaluation
            (accuracy, confusion_matrix) = evaluate(validation_db, trained_tree)
            # prune
            pruned_tree = prune_tree(trained_tree, validation_db, accuracy)
            # evaluate on now pruned tree
            (pruned_accuracy, pruned_confusion_matrix) = evaluate(test_db, pruned_tree)
            pruned_accuracies.append(pruned_accuracy)
            agg_pruned_confusion_matrix += pruned_confusion_matrix
    agg_pruned_confusion_matrix /= (k * k - 1)
    calculate_measures(agg_pruned_confusion_matrix)
    avg_pruned_accuracy = np.average(pruned_accuracies)

    print("Average Accuracy: ", average_accuracy)
    print("Average Accuracy of Pruned Decision Tree: ", avg_pruned_accuracy)


if __name__ == "__main__":
    main(sys.argv[-1])
