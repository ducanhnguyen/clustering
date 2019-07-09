'''
Implementation of soft-kmeans
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class Soft_K_means:

    def fit(self, X, K=5, beta=3):
        """

        :param X:
        :param K: number of clusters
        :param beta:
        :return:
        """
        N = len(X)

        # just for testing
        fig = plt.figure(figsize=(20, 15))
        columns = 4
        rows = 5

        # initialize mean of clusters
        M = X[np.random.choice(len(X), size=K)]
        R = np.zeros(shape=(N, K))

        # calculate mean of clusters
        costs = []
        iterations = []
        converge = False
        iteration = -1

        while not converge:
            iteration += 1
            iterations.append(len(iterations))

            # step 1: calculate cluster responsibilities
            for n in range(N):
                for k in range(K):
                    R[n][k] = np.exp(- beta * self.diff(X[n], M[k]))

            R /= R.sum(axis=1, keepdims=True)

            # step 2: recalculate mean of clusters
            for k in range(K):
                numerator = 0
                denominator = 0

                for n in range(N):
                    numerator += R[n, k] * X[n]
                    denominator += R[n, k]

                M[k] = numerator / denominator

            # compute cost
            costs.append(self.compute_cost(X, M, R))

            # check the convergence of the algorithm
            if len(costs) >=2 and np.abs(costs[-1] - costs[-2]) == 0:
                converge = True

            # just for testing
            if (iteration % 10 == 0 or converge):
                fig.add_subplot(rows, columns, np.ceil(iteration / 10) + 1)
                plt.scatter(X[:, 0], X[:, 1])
                plt.scatter(M[:, 0], M[:, 1])
                plt.title('iteration ' + str(iteration))

        # just for testing
        plt.show()

        plt.plot(iterations, costs)
        plt.title('Cost of soft-kmeans over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.show()

    def diff(self, x, y):
        return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def compute_cost(self, X, M, R):
        cost = 0
        for n in range(len(X)):
            for k in range(len(M)):
                cost += self.diff(X[n], M[k]) * R[n, k]
        return cost


if __name__ == '__main__':
    # generate samples for clustering
    X, _ = make_blobs(n_samples=150, centers=5, random_state=100)
    s = Soft_K_means()
    s.fit(X)
