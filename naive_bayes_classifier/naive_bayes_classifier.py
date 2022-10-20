import pandas as pd
import numpy as np
# from utils.functions import standard_deviation


class naive_bayes_classifier:
    def __init__(self):
        #                                                                                        attrb 1      attrb 2
        # will save the mean and standard deviation of each attribute for each class: class 1 [(mean, std), (mean, std), ...]
        #                                                                             class 2 [    ...    ,     ...    , ...]
        self.__attribute_matrix_mean_std = pd.DataFrame([[]])
        # will save the priori for each class
        self.__priori = []  # [<priori_class_1>, <priori_class_2>, ...]

    def fit(self, X, y):
        classes = pd.Series(y.unique())
        values = [[() for _ in range(len(X.columns))] for _ in range(len(classes))]
        self.__attribute_matrix_mean_std = pd.DataFrame(values, columns=X.columns, index=classes)

        for cls in self.__attribute_matrix_mean_std.index.values:
            i_cls_samples = X.index[np.where(y == cls)] # this take the index returned for np and take the pd (X) index
            Xc = X.loc[i_cls_samples]
            self.__priori.append(len(Xc) / len(X))  # insert priori for each group of samples separated per class

            for feature in X.columns:
                column_mean = np.mean(Xc[feature])
                column_std = np.std(Xc[feature])
                self.__attribute_matrix_mean_std.loc[cls, feature] = (column_mean, column_std)
    
    def predict(self, x):
        if len(x) != len(self.__attribute_matrix_mean_std.columns):
            raise Exception('Sample invalid')

        posteriors = [] # [[<posteriori>, <class>], ...]

        for i_cls, cls in enumerate(self.__attribute_matrix_mean_std.index.values):
            likelihood = 1
            for i_feature, feature in enumerate(self.__attribute_matrix_mean_std.columns):
                column_mean = self.__attribute_matrix_mean_std.loc[cls, feature][0]
                column_std = self.__attribute_matrix_mean_std.loc[cls, feature][1]
                if column_std == 0:
                    likelihood = likelihood * self.__gaussian(x[i_feature], column_mean, 1)
                else:
                    likelihood = likelihood * self.__gaussian(x[i_feature], column_mean, column_std)

            posteriors.append([
                likelihood * self.__priori[i_cls], 
                cls
            ])

        greater_posteriori_class = sorted(
            posteriors,
            key=lambda posteriors: posteriors[0] # sorted by posteriori
        )[-1][1]

        return greater_posteriori_class

    def __gaussian(self, x, mu, sig):
        prob = (1 / (np.sqrt(2 * np.pi) * sig)) * np.exp((-1 / 2) * ((x - mu) / sig)**2)
        return prob

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
