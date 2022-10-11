import pandas as pd
import numpy as np
# from utils.functions import standard_deviation


class naive_bayes_classifier:
    def __init__(self):
        # will save the class probabilities of each class for each sample
        self.__class_matrix_prob = pd.DataFrame([])

    def fit(self, X, y):
        # initializing the class_matrix_prob
        classes = pd.Series(y.unique())
        values = [[ 0 for _ in range(len(classes)) ] for _ in range(len(X))]
        self.__class_matrix_prob = pd.DataFrame(values, columns=classes)

        for cls in self.__class_matrix_prob.columns:
            i_cls_samples = np.where(y == cls)
            Xc = X.loc[i_cls_samples]

            for i_sample, x in enumerate(X.values):
                likelyhood = self.__likelyhood(x, Xc)
                priori = len(Xc)/len(X)
                self.__class_matrix_prob.loc[i_sample, cls] = likelyhood * priori

        print(self.__class_matrix_prob)
    
    def __likelyhood(self, x, Xc):
        lkh = 1

        for i_feature, feature in enumerate(Xc.columns):
            column_mean = np.mean(Xc[feature])
            column_std = np.std(Xc[feature])
            lkh = lkh * self.__gaussian(x[i_feature], column_mean, column_std)

        return lkh

    def __gaussian(self, x, mu, sig):
        prob = (1/np.sqrt(2*np.pi*sig)) * np.exp((-1/2) * ((x-mu)/sig)**2)
        return prob

    def predict(self, x):
        pass
