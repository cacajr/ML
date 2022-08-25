from statistics import mode
import pandas as pd
import numpy as np


class knn:
    def __init__(self, k=3, type_distance_calc = 'euclidean'):
        self.__k = k
        self.__type_distance_calc = type_distance_calc
        self.__X = pd.DataFrame([])
        self.__y = pd.Series([])

        self.__distances = []   # [[<index_sample>, <distance_value>], ...]
    
    def fit(self, X, y):
        self.__X, self.__y = X, y

    def predict(self, x):
        if len(x) != self.__X.columns.size:
            raise Exception('Sample invalid')

        self.__distances = [] # refresh list

        for index, sample in enumerate(self.__X.values):
            self.__distances.append(
                [index, self.__distance_calc(x, sample)]
            )

        self.__distances = sorted(
            self.__distances, 
            key=lambda distance: distance[1] # sorted by distance
        )

        return self.__choose_closest_sample()

    def __distance_calc(self, coord1, coord2):
        if self.__type_distance_calc == 'euclidean':
            return self.__euclidean_distance(coord1 - coord2)

        raise Exception('Distance not implemented')

    def __euclidean_distance(coord1, coord2):
        coord1, coord2 = np.array(coord1), np.array(coord2)

        distance = np.sqrt(
            np.sum(
                np.power((coord1 - coord2), 2)
            )
        )

        return distance

    def __choose_closest_sample(self):
        candidates = self.__distances[:self.__k]
        candidates_classes = [
            self.__y.values[candidate[0]] 
            for candidate in candidates
        ]

        return mode(candidates_classes)

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
