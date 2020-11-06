# Decision Trees - Coursework 1

This project focuses on developing a well-evaluated method of training decision trees to high accuracy on WiFi localization datasets (included in `wifidb/`).

## Running Training & Evaluation

Requirements are `numpy` and `matplotlib` (outlined in requirements.txt) and Python 3 (we recommend 3.8).

The script takes a single parameter (a commandline argument, the path for the dataset to use).

The syntax is as follows:
``` bash
python main.py path/to/dataset.txt
```

To run training & evaluation on both datasets, use:
``` bash
python main.py wifidb/clean_dataset.txt
python main.py wifidb/noisy_dataset.txt
```

## Training (programmatic)

The training utility is provided in `train.py`.

This can be accessed programmatically (separately from the full pipeline) via the `train` function.

The train function's exact nature is described further in `train.py` -- in summary, it requires simply the `numpy` parsed dataset, and will return both the tree and its total depth.

Example:
``` python
from train import train

data = np.loadtxt('path/to/my/dataset.txt')
tree, depth = train(data)
```

## Prediction (programmatic)

The prediction utility is provided in `evaluate.py`.

This can be accessed programmatically via the `predict` function, which is described in `evaluate.py` -- in summary, it requires the input features (wifi signals) and a tree, and will return the predicted Room.

Example:
``` python
from evaluate import predict
...
tree = ...
X = ...
Y = predict(X, tree)
```

## Evaluation (programmatic)

The evaluation utility is provided in `evaluate.py`.

This can be accessed programmatically via the `evaluate` and `calculate_measures` functions.

These are described in `evaluate.py`. In summary, `evaluate` requires a testing dataset and the trained tree as input, and will output both the accuracy and confusion matrix. `calculate_measures` simply requires the confusion matrix, and will print out relevant metrics.

Example:
``` python
from evaluate import evaluate calculate_measures
...
tree = ...
test_data = ...
accuracy, confusion_matrix = evaluate(test_data, tree)
calculate_measures(confusion_matrix) # print out relevant metrics, e.g. F1, recall, precision
```

## Compatibility

This repository only has two requirements (`numpy` and `matplotlib`) and as a result will work on most environments. We've tested it on:

- MacOS 10 and 11
- Windows 10
- Ubuntu 16 and 18

We highly recommend using a Linux environment such as Ubuntu when running this repository. The CI/CD pipeline configured with this repository utilizes an Ubuntu environment when linting and running `main.py` for training and evaluation on both datasets.
