import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import random


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

def plot_column_histogram_and_distribution(column, feature_name, distr_name):
    distribution = getattr(scipy.stats, distr_name)
    distri_params = distribution.fit(column)
    new_distr = distribution(*distri_params)

    # generate a range(0, 1, 100) numbers
    x = np.linspace(0, 1, 100)
    _, ax = plt.subplots(1, 2)

    plt.suptitle(feature_name + ' - ' + distr_name)
    sns.histplot(x=column, ax=ax[0])
    sns.lineplot(x=x, y=new_distr.pdf(x), ax=ax[1])
    plt.show()

def create_samples_rejection_method(X, y, qtd_per_class={}, distributions=[], distributions_method=[]):
    if len(qtd_per_class.keys()) == 0:
        qtd_per_class = {key: 10 for key in y.unique()}
    elif len(qtd_per_class.keys()) != len(y.unique()):
        raise Exception('Quantity per class size is different number of classes')

    if len(distributions) == 0:
        distributions = ['norm' for _ in range(len(X.columns))]
    elif len(distributions) != len(X.columns):
        raise Exception('Distributions size is different X columns size')

    if len(distributions_method) == 0:
        distributions_method = ['norm' for _ in range(len(X.columns))]
    elif len(distributions_method) != len(X.columns):
        raise Exception('Distributions method size is different X columns size')

    new_X = X.copy()
    new_y = y.copy()
    classes = y.unique()

    for cls in classes:
        i_cls_samples = new_X.index[np.where(y == cls)] # this take the index returned for np and take the pd (X) index
        Xc = new_X.loc[i_cls_samples] 

        for _ in range(qtd_per_class[cls]):
            new_sample = []

            for i_feature, feature in enumerate(new_X.columns):
                distribution = getattr(scipy.stats, distributions[i_feature])
                distr_params = distribution.fit(Xc[feature])
                new_distr = distribution(*distr_params)

                distribution_method = getattr(scipy.stats, distributions_method[i_feature])
                distr_params_method = distribution_method.fit(Xc[feature])
                new_distr_method = distribution_method(*distr_params_method)

                while True:
                    rand_value = random.randint(0, 100) / 100
                    new_sample_attribute = random.randint(0, 100) / 100

                    if new_distr_method.pdf(new_sample_attribute) * rand_value < new_distr.pdf(new_sample_attribute):
                        new_sample.append(new_sample_attribute)
                        break

            new_X.loc[new_X.index.size] = new_sample
            new_y.loc[new_y.size] = cls

    return new_X, new_y

def remove_values_randomly(X, qtd_values=10, value_replace='?'):
    new_X = X.copy()
    columns = new_X.columns

    for _ in range(qtd_values):
        i_sample = random.randint(0, new_X.index.size - 1)
        i_attribute = random.randint(0, new_X.columns.size - 1)

        
        while new_X.loc[i_sample, columns[i_attribute]] == value_replace:
            i_sample = random.randint(0, new_X.index.size - 1)
            i_attribute = random.randint(0, new_X.columns.size - 1)

        new_X.loc[i_sample, columns[i_attribute]] = value_replace

    return new_X
