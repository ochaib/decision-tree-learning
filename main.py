import sys
import numpy as np
from train import train
from evaluate import evaluate, calculate_measures, prune_tree

def generate_test_training(dataset, k):
    # Shuffle test dataset
    np.random.shuffle(dataset)

    # Divide the dataset into k equal folds/splits.
    folds = np.array_split(dataset, k)

    # Use k-1 (9) folds for training+validation and 1 for testing
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
    training_sets, test_sets = generate_test_training(np_dataset, k)
    agg_confusion_matrix = np.zeros((4, 4))
    for i in range(k):
        training_db = training_sets[i]
        test_db = test_sets[i]
        #* train
        trained_tree, depth = train(training_db, 1)
        
        #* evaluate
        (accuracy, confusion_matrix) = evaluate(test_db, trained_tree)
        agg_confusion_matrix += confusion_matrix
        accuracies.append(accuracy)
    #? Calculate average accuracy?
    #? Or just select one with highest accuracy.
    agg_confusion_matrix /= k
    calculate_measures(agg_confusion_matrix)
    average_accuracy = np.average(accuracies)
    print("Average Accuracy: ", average_accuracy)

if __name__ == "__main__":
    main(sys.argv[-1])