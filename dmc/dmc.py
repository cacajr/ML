import pandas as pd
import numpy as np


class dmc:
    def __init__(self, type_distance_calc = 'euclidean'):
        self.__type_distance_calc = type_distance_calc

        self.__centroids = [] # [{'centroid': <list_centroid>, 'target': <target_centroid>}, ...]

    def fit(self, X, y):
        classes = pd.Series(y.unique())

        for cls in classes:
            i_cls_samples = X.index[np.where(y == cls)] # this take the index returned for np and take the pd (X) index
            Xc = X.loc[i_cls_samples]

            centroid = []
            for feature in Xc.columns:
                centroid.append(Xc[feature].mean())

            self.__centroids.append({
                'centroid': centroid,
                'target': cls,
            })

    def predict(self, x):
        if len(x) != len(self.__centroids[0]['centroid']):
            raise Exception('Sample invalid')

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
            return self.__euclidean_distance(coord1, coord2)

        raise Exception('Distance not implemented')

    def __euclidean_distance(self, coord1, coord2):
        coord1, coord2 = np.array(coord1), np.array(coord2)

        distance = np.sqrt(
            np.sum(
                np.power((coord1 - coord2), 2)
            )
        )

        return distance

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
