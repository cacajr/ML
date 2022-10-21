import pandas as pd
import numpy as np


class bayes_classifier:
    def __init__(self):
        # The covariance matrix list will save one covariance matrix to each class set.
        self.__covariance_matrix_list = []  #[[<covariance_matrix_class_1>], [<covariance_matrix_class_2>], ...]

        # The mean list will save a array with mean of each column to each class set
        self.__mean_list = []   #[[<mean_class_1_for_each_attribute>], [<mean_class_2_for_each_attribute>], ...]
        
        # The class array will save the attributes/features
        self.__feature_array = []
        # The class array will save the classes/target
        self.__class_array = []

        # The i will save a matrix with diagonal with values equel to 1 * noise on the main diagonal and 0 on the rest
        self.__noise_matrix = [[]]

    def fit(self, X, y):
        self.__class_array = y.unique()
        self.__feature_array = X.columns

        self.__noise_matrix = np.eye(len(self.__feature_array)) * 1e-10    # creating the noise matrix

        for cls in self.__class_array:
            i_cls_samples = X.index[np.where(y == cls)] # this take the index returned for np and take the pd (X) index
            Xc = X.loc[i_cls_samples]
            self.__covariance_matrix_list.append(pd.DataFrame(np.cov(Xc, rowvar=False)))
            self.__mean_list.append(np.array([Xc[feat].mean() for feat in self.__feature_array]))
    
    def predict(self, x):
        if len(x) != len(self.__feature_array):
            raise Exception('Sample invalid')

        posteriors = [] # [[<posteriori>, <class>], ...]

        for i_cls, cls in enumerate(self.__class_array):
            prob = self.__multivariate_gaussian(x, self.__mean_list[i_cls], self.__covariance_matrix_list[i_cls])

            posteriors.append([
                prob,
                cls
            ])

        greater_posteriori_class = sorted(
            posteriors,
            key=lambda posteriors: posteriors[0] # sorted by posteriori/prob
        )[-1][1]

        return greater_posteriori_class

    def __multivariate_gaussian(self, x, mu, sig):
        det_sig = np.linalg.det(sig)

        if det_sig == 0:    # if det covariance matrix is equal 0, so i use det equal 1 and add noise
            det_sig = 1
            sig = sig + self.__noise_matrix

        inv_sig = np.linalg.inv(sig)

        prob = 1 / (np.power(2 * np.pi, len(x) / 2) * np.power(det_sig, 1 / 2)) * np.exp((-1 / 2) * np.dot(np.dot((x - mu), inv_sig), (x - mu)))
        
        return prob

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
