'''
Raw implementation of Gaussian mixture model without optimizations.

Tutorial: http://cs229.stanford.edu/notes/cs229-notes7b.pdf

'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    '''
    Gaussian mixture model
    '''

    def __init__(self, K):
        '''

        :param K: the number of gaussian distributions
        '''
        self.K = K

    def fit(self, X, max_iterations=1000, smoothing=1e-2):
        '''
        Train gaussian mixture model
        :param X: dataset
        :param max_iterations: the number of iterations to update mean, pi, and covariance matrix until it is converged.
        :param smoothing: add smoothing to covariance to avoid zero standard deviation.
        :return: the estimated mean, pi, and covariance matrix
        '''
        M, D = X.shape
        mean = np.random.randn(self.K, D)  # mean vector
        pi = np.ones(shape=(self.K, 1)) / self.K
        xichma = np.zeros(shape=(self.K, D, D))  # covariance matrix

        likelihoods = []

        # Initialize covariance matrix
        for j in range(self.K):
            for idx in range(D):
                xichma[j, idx, idx] = 1

        # train GMM
        for iteration in range(max_iterations):

            # E-step
            W = np.zeros(shape=(M, self.K))

            for j in range(self.K):
                for i in range(M):
                    W[i, j] = multivariate_normal.pdf(x=X[i], mean=mean[j], cov=xichma[j]) * pi[j]

            for i in range(M):
                W[i, :] = W[i, :] / np.sum(W[i])

            # M-step
            for j in range(self.K):
                tmp = np.sum(W[:, j])

                # re-estimate pi
                pi[j] = 1 / M * tmp

                # re-estimate mu
                mean[j] = np.zeros(shape=(D,))
                for i in range(M):
                    mean[j] += W[i, j] * X[i]
                mean[j] /= tmp

                # re-estimate covariance matrix
                xichma[j] = np.zeros(shape=(D, D))

                for i in range(M):
                    V = (X[i] - mean[j]).reshape(-1, 1)  # (D, 1)
                    U = np.matmul(V, np.transpose(V))  # (D, D)
                    xichma[j] += W[i, j] * U  # (D, D)

                xichma[j] /= tmp

                # Add smoothing to avoid zero standard deviation, which leads to division by zero when compute pdf.
                xichma += np.eye(D) * smoothing

            # compute loss
            likelihood = self.compute_likelihood(X, self.K, pi, xichma, mean)
            likelihoods.append(likelihood)
            print(
                'iteration ' + str(iteration) + ' / loss = ' + str(likelihood) + ' / mean = ' + str(mean).replace('\n',
                                                                                                                  ' '))

            # check convergence
            if len(likelihoods) >= 2 and np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-6:
                print('Converged! Exit.')
                break

        # save
        self.mean = mean
        self.pi = pi
        self.xichma = xichma
        self.likelihoods = likelihoods
        self.W = W
        self.X = X

    def compute_likelihood(self, X, K, pi, xichma, mean):
        loss = 0

        M = len(X)
        for i in range(M):

            inner = 0
            for j in range(K):
                inner += multivariate_normal.pdf(x=X[i], mean=mean[j], cov=xichma[j]) * pi[j]

            loss += np.log(inner)

        return loss

    def plot_dataset(self):
        '''
        Plot dataset after applying GMM
        :return:
        '''
        if (self.K == 3):
            M = len(self.X)

            for i in range(M):
                plt.scatter(X[i, 0], X[i, 1], c=np.array([self.W[i, :]]))

            plt.title('Clustering by GMM')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
        else:
            print('Do not support when K != 3')

    def plot_likelihood(self):
        plt.plot(self.likelihoods)
        plt.title('Likelihood of GMM over iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Likelihood')
        plt.show()


def create_dataset():
    mu1 = np.array([1, 2])
    Z1 = np.random.randn(200, 2) + mu1

    mu2 = np.array([4, 2])
    Z2 = np.random.randn(200, 2) + mu2

    mu3 = np.array([6, 7])
    Z3 = np.random.randn(200, 2) + mu3

    X = np.concatenate((Z1, Z2, Z3), axis=0)
    plt.scatter(Z1[:, 0], Z1[:, 1], c='red')
    plt.scatter(Z2[:, 0], Z2[:, 1], c='blue')
    plt.scatter(Z3[:, 0], Z3[:, 1], c='green')
    plt.title('Dataset before GMM')
    plt.show()

    return X


if __name__ == '__main__':
    # create a dataset which is composed of 3 gaussian distributions
    X = create_dataset()

    # Initialize GMM with a specified number of gaussian mixture models (K). Here, I set K = 3.
    # You can try other values of K.
    gmm = GMM(K=3)

    # Train GMM to find the parameters of gaussian mixture models.
    gmm.fit(X)

    # Let see the results of mean, covariance matrix; and compare them with the true values of these parameters.
    print('final mu = ' + str(gmm.mean))
    print('final pi = ' + str(gmm.pi))
    print('final xich ma = ' + str(gmm.xichma))

    # Plot likelihood over iterations. As you can see, the value of likelihood increases.
    gmm.plot_likelihood()
    gmm.plot_dataset()
