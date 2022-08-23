import pandas as pd
import numpy as np


class dmc:
    def __init__(self, type_distance_calc = 'euclidean'):
        self.__type_distance_calc = type_distance_calc

        self.__centroids = [] # [{'centroid': <list_centroid>, 'target': <target_centroid>}, ...]

    def fit(self, X, y):
        targets = y.unique()

        Xy = pd.concat([X, y], axis=1)

        for target in targets:
            X_target = Xy[Xy[Xy.columns[-1]].isin([target])]   # return DataFrame with the samples with the especific target

            centroid = []
            for column in X.columns:
                centroid.append(X_target[column].mean())

            self.__centroids.append({
                'centroid': centroid,
                'target': target,
            })

    def predict(self, x):
        distance = []   # [[<distance>, {'centroid': <list_centroid>, 'target': <target_centroid>}], ...]

        for centroid in self.__centroids:
            distance.append([
                self.__distance_calc(x, centroid['centroid']),
                centroid
            ])

        return sorted(
            distance,
            key=lambda distance: distance[0] # sorted by distance
        )[0][1]['target']

    def __distance_calc(self, coord1, coord2):
        if self.__type_distance_calc == 'euclidean':
            return np.linalg.norm(coord1 - coord2)

        raise Exception('Distance not implemented')

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
