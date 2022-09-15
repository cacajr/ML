import pandas as pd
import numpy as np


class kmeans:
    def __init__(self, k=3, type_distance_calc = 'euclidean', number_iteration = 300):
        self.__k = k
        self.__type_distance_calc = type_distance_calc
        self.__number_iteration = number_iteration

        self.__X = pd.DataFrame([])

        self.__centroids = pd.DataFrame([])
        self.__clusters = []    # [pd.DataFrame(), ...]
        self.__centroids_targets = []
    
    def fit(self, X, y=[]):
        self.__X = X    # save the data in memory

        self.__centroids = self.__X.sample(n=self.__k)  # choose the centroids

        self.__clusters = [    # separating one cluster for each centroid
            pd.DataFrame([], columns=self.__X.columns)
            for _ in self.__centroids.values
        ]

        for i in range(self.__number_iteration):
            for index_sample, sample in zip(self.__X.index.values, self.__X.values):  # insert a sample in your cluster based on centroid
                sample_to_centroid_distances = [] # [{'index_centroid': 0, 'distance': 0}, ...]

                for index_centroid, centroid in enumerate(self.__centroids.values):    # calculating distance
                    sample_to_centroid_distances.append({
                        'index_centroid': index_centroid,
                        'distance': self.__distance_calc(sample, centroid)
                    })

                shorter_sample_to_centroid_distance = min(  # find shorter distance
                    sample_to_centroid_distances, 
                    key=lambda dict: dict['distance']
                )

                self.__clusters[    # insert sample in corresponding cluster
                    shorter_sample_to_centroid_distance['index_centroid']
                ].loc[index_sample] = sample

            # update centroids ...
            self.__centroids = pd.DataFrame([], columns=self.__centroids.columns)

            for cluster in self.__clusters:
                new_centroid = []
                for column in cluster.columns:
                    new_centroid.append(cluster[column].mean())

                self.__centroids.loc[len(self.__centroids.index)] = new_centroid

            # update centroids targets ...
            if len(y) != 0:
                for cluster in self.__clusters:
                    self.__centroids_targets.append(
                        y.loc[cluster.index.values].groupby(
                            y.loc[cluster.index.values]
                        ).count().sort_values().index[-1]   # return the target most common in cluster
                    )

            if i < self.__number_iteration - 1: # do not reset cluster in the last iteration
                # reset clusters ...
                self.__clusters = [    # separating one cluster for each centroid
                    pd.DataFrame([], columns=self.__X.columns)
                    for _ in self.__centroids.values
                ]

    def predict(self, x):
        if len(x) != len(self.__centroids.values[0]):
            raise Exception('Sample invalid')

        distance = []   # [[<distance>, <index_centroid>], ...]

        for index, centroid in enumerate(self.__centroids.values):
            distance.append([
                self.__distance_calc(x, centroid),
                index
            ])

        closest_centroid_index = sorted(
            distance,
            key=lambda distance: distance[0] # sorted by distance
        )[0][1]

        return self.__centroids_targets[closest_centroid_index]

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

    def get_clusters(self):
        return self.__clusters

    def score(self, X_test, y_test):
        hits = 0

        for sample, predict in zip(X_test.values, y_test.values):
            if self.predict(sample) == predict:
                hits += 1

        return hits/y_test.size
