import pandas as pd
import numpy as np


def min_max_normalization(X):
    X_norm = X.copy()

    for column in X_norm.columns:
        X_norm[column] = (X_norm[column] - np.min(X_norm[column])) / (np.max(X_norm[column]) - np.min(X_norm[column]))

    return X_norm

def confusion_matrix(y, y_pred):
    y = pd.Series(y)
    classes = pd.Series(y.unique())
    confusion_matrix = [
        [
            0 
            for _ in range(len(classes))  # for each class, adding one column in matrix
        ] 
        for _ in range(len(classes))  # for each class, adding one line in matrix
    ]

    for cls, pred in zip(y, y_pred):
        if cls == pred:
            line = classes.index[classes == cls][0]   # take the index real class
            col = line  # if pred is equals cls, so repeat the index to predict
            confusion_matrix[line][col] += 1
        else:
            line = classes.index[classes == cls][0]   # take the index real class
            col = classes.index[classes == pred][0]   # take the index predict class
            confusion_matrix[line][col] += 1

    return pd.DataFrame(confusion_matrix, columns=classes, index=classes)

def confusion_matrix_mean(list):
    confusion_matrix_mean = list[0].copy()

    for matrix in list[1:]:
        confusion_matrix_mean += pd.DataFrame(matrix)

    confusion_matrix_mean //= len(list)

    return confusion_matrix_mean

def standard_deviation(list):
    return np.sqrt(
        np.sum(
            np.power(
                np.array(list) - (np.sum(list)/len(list)), 
                2
            )
        ) / len(list)
    )