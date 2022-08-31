import pandas as pd


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

    return confusion_matrix
