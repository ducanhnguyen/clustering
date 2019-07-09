'''
Just an example of hierarchical clustering.

Dataset: Iris

'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

class Hierarchical_Clustering:

    def load_data(self, path= '../data/iris/Iris.csv'):
        data = pd.read_csv(path)
        X = data.drop(labels=['Id', 'Species'], axis=1).to_numpy()
        return X

    def get_clusters(self, Z):
        clustering = dict()
        clustering[1] = set()
        clustering[2] = set()
        clustering[3] = set()

        clustering[1].add(Z[0, 0])

        for j in range(10):
            for i in range(1, len(Z)):
                observation_1 = Z[i, 0]
                observation_2 = Z[i, 1]
                dist = Z[i, 2]
                total_observations = Z[i, 3]

                if observation_1 in clustering[1]:
                    clustering[1].add(observation_2)

                if observation_2 in clustering[1]:
                    clustering[1].add(observation_1)

        print(clustering[1])

    def single_linkage(self,X):
        '''
        Distance between two clusters = the min distance between two data points
        :param X:
        :return:
        '''
        linked = linkage(X, 'single')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Single linkage')
        plt.xlabel('Data point index')
        plt.ylabel('Height')
        plt.show()

        return linked

    def complete_linkage(self,X):
        '''
        Distance between two clusters = the max distance between two data points
        :param X:
        :return:
        '''
        linked = linkage(X, 'complete')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Complete linkage')
        plt.xlabel('Data point index')
        plt.ylabel('Height')
        #plt.show()

        return linked

    def mean(self,X):
        '''
        Distance between two clusters = the average distance
        :param X:
        :return:
        '''
        linked = linkage(X, 'average')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Mean')
        plt.xlabel('Data point index')
        plt.ylabel('Height')
        plt.show()

        return linked

    def ward(self,X):
        linked = linkage(X, 'ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked)
        plt.title('Ward')
        plt.xlabel('Data point index')
        plt.ylabel('Height')
        plt.show()

        return linked


if __name__ == '__main__':
    clustering = Hierarchical_Clustering()
    X = clustering.load_data()
    Z = clustering.complete_linkage(X)
    clustering.get_clusters(Z)

