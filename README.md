# VIC-ValidityIndex
VIC [1] is a Cluster Validation technique that uses a set of supervised classifiers to select which clustering algorithm to apply for a given problem.

This implementation was developed in python, and uses a custom dataset of human fingerprints represented as minutiae for fingerprint recognition. For more information on the data set, please check `Assignment 2.pdf`.

VIC uses k-folds cross validation to train the ensemble of supervised classifiers on each partition of the dataset. Once the classifiers are trained, the mean ROC AUC of all the k-folds is used as an indicator of the validity of each classifier. In the end, the highest AUC value among all partitions indicates the most optimal partition of the data. The algorithm works as follows:


    INPUT dataset, classifier s, kFolds, nPartitions;
    OUTPUT report;

    report = [];
    features, partitions = splitDataset( dataset, partitions );

    FOR EACH p IN nPartitions:
        partition = partitions[ p ];
        classifierMeans = [];
        FOR EACH c IN classifiers:
            AUCs = [];
            FOR EACH k IN kFolds:
                trainX, trainY, testX, testY = kFoldSplit( features, partition );
                trainClassifier( c, trainX, trainY );
                AUCs[ k ] = calculateAUC( c, testX, testY );
            meanAUC = mean( AUCs );
            classifierMeans[ c ] = meanAUC;
        report[ p ] = [ p, classifierMeans, max( classifierMeans ) ];

## Available Classifiers

The current implementation supports 7 classifiers:

1. MultiLayerPerceptron
2. Adaboost
3. K-Nearest Neighbors
4. Random Forest
5. Decision Trees
6. Naive Bayes
7. Linear Discriminant Analysis

## Adding/Removing Classifiers

Adding new classifiers is as simple as adding an `elif` statement in the `get_estimator` function from line 41 on `evaluate_classifiers.py` following the existing examples present. For example:

    ...
    elif( est_type == 'NewCustomClassifier' ):
        estimator = NewCustomClassifier()
    ...

Please notice that in order the new classifier work properly, it must at least inherit from the class `klearn.base.BaseEstimator`, and follow the guidelines described by sklearn [here](https://scikit-learn.org/stable/developers/develop.html).

The final step is to add the classifiers to the `-c` argument on line `26`, and call `evaluate_classifiers.py` with the proper values.

## Usage

To test the current implementation all you need to is:

1. Run the command `python evaluate_classifiers.py -i <Path_to_my_csv>` in a terminal .
2. Optionally you can specify any subset of classfiers to use by passing the argument `-c Classifier1,Classifier2`.
3. Optionally you can specify the number of folds for the CV by passing the argument `-k <NumberFolds>`.

The script will generate a text file named `evaluation_<k>_fold.txt` with the results for all the partitions.

When running on a different dataset, make sure to add a column for all the different partitions that you want to evaluate on the rightmost side of your dataset as follows:

Feature 1 | ... | Feature N | Partition 1 | ... | Partition P |
--------- | --- | --------- | ----------- | --- | ----------- |
... | ... | ... | ... | ... | ... |

The algorithm will take care of splitting the partitions and choosing the right one every iteration.

## Authors

* Laura Pérez - ([https://github.com/LauraJaideny](https://github.com/LauraJaideny))
* Andreé Vela - ([https://github.com/AndreeVela/](https://github.com/AndreeVela/))
* Juan Carlos Ángeles - ([https://github.com/jcarlosangelesc](https://github.com/jcarlosangelesc))

## Bibliography

[1] J. Rodríguez, M. A. Medina-Pérez, A. E. Gutierrez-Rodríguez, R. Monroy, H. Terashima-Marín. [Cluster validation using an ensemble of supervised classifiers](https://www.sciencedirect.com/science/article/abs/pii/S0950705118300091). *Knowledge-Based Systems*, Volume 145 (2018). Pages 134-144.
