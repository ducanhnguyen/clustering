'''
This file contains these following parts:

- Given a dataset, choose the best number of clusters based on inertia values and silhouette scores
- Given the selected number of clusters, make a word cloud
- Given a word in the word cloud, find the phrases containing this word

'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud, STOPWORDS

TITLE_COLUMN = 'title'
ABSTRACT_COLUMN = 'abstract'
KEYWORD_COLUMN = 'keywords'
GROUP_COLUMN = 'groups'


def preprocess_data(data_train):
    '''

    :param data_train: A dataframe
    :return: new dataframe with additional columns
    '''
    # get all distinct research groups
    groups = data_train[GROUP_COLUMN].dropna().str.split('\n')

    distinct_groups = set()
    for group in groups:
        for item in group:
            distinct_groups.add(item)

    # add new columns to data which are research groups
    for distinct_group in distinct_groups:
        data_train[distinct_group] = 0

    for row_idx in range(len(data_train)):
        row_content = data_train.at[row_idx, GROUP_COLUMN]
        if not pd.isna(row_content):  # some rows missed value should be ignored
            groups = row_content.split('\n')

            for distinct_group in distinct_groups:
                if distinct_group in groups:
                    data_train.at[row_idx, distinct_group] = 1

    return data_train


def choose_the_best_clustering(X, max_num_clusters=30, displayed=True):
    '''
    Choose the best clustering
    :param X: is a N by M matrix, where N is the number of points. Each point is a M-dimensional vector.
    :param max_num_clusters: the maximum number of clusters
    :return:
    '''
    inertia_values = []
    num_clusters = []
    silhouette_values = []

    for num_cluster in range(2, max_num_clusters):  # ignore the case that the number of clusters = 1
        num_clusters.append(num_cluster)

        # Given the number of clusters, there are more than one results of clustering.
        # kmeans automatically chooses the best clustering based on the value of inertia.
        kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=20).fit(X)
        clustering = kmeans.fit(X)

        inertia = compute_inertia(clustering.cluster_centers_, X)
        inertia_values.append(inertia)

        # silhouette_score only works in case of more than 1 clusters
        silhouette = silhouette_score(X, clustering.labels_)
        silhouette_values.append(silhouette)

        print(str(num_cluster) + ' clusters / inertia = ' + str(inertia) + ' / silhouette score = ' + str(silhouette))

    # plot to choose the best number of clusters manually
    if displayed:
        plt.plot(num_clusters, inertia_values, 'b*-', color='blue', label='inertia')
        plt.plot(num_clusters, silhouette_values, 'b*-', color='black', label='silhouette')
        plt.title('Choose the best number of clusters')
        plt.xlabel('The number of clusters')
        plt.ylabel('Value')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


def diff(x, y):
    '''
    Distance between two N-dimensional points
    :param x: point 1
    :param y: point 2
    :return:
    '''
    return np.sqrt(np.sum((x - y) ** 2))


def compute_inertia(centroids, X):
    '''
    Compute inertia
    :param centroids: a list of centroids
    :param X: is a N by M matrix, where N is the number of points. Each point is a M-dimensional vector.
    :return: a scalar value
    '''
    inertia = 0

    for idx in range(len(X)):
        # need to compute the distance between a point and its closest centroid
        inertia += np.min([diff(X[idx], centroid) for centroid in centroids])

    inertia /= len(X)
    return inertia


def plot_data_in_2d(X, labels, num_cluster):
    colors = np.random.rand(num_cluster)

    X_embedded = PCA(n_components=2).fit_transform(X)
    # X_embedded = TSNE(n_components=2, n_iter=10000).fit_transform(X)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors[labels])
    for i in range(len(labels)):
        plt.annotate(labels[i], [X_embedded[i, 0], X_embedded[i, 1]])

    plt.title('Clustering result on 2D')
    plt.show()


def generate_word_cloud(labels, data_train, num_cluster):
    '''

    :param labels: 1-D dimensional array
    :param data_train: a dataframe
    :return:
    '''
    fig = plt.figure(figsize=(15, 15))
    columns = 4
    rows = np.ceil(num_cluster / columns) + 1

    for cluster in range(num_cluster):  # ignore the case that the number of clusters = 1
        txt = ''

        for idx in range(len(labels)):

            cluster_index = labels[idx]
            if cluster_index == cluster:
                txt += data_train.at[idx, TITLE_COLUMN] + ' '
                txt += data_train.at[idx, ABSTRACT_COLUMN] + ' '
                txt += data_train.at[idx, KEYWORD_COLUMN] + ' '

        # create word cloud
        fig.add_subplot(rows, columns, cluster + 1)
        wordcloud = WordCloud(stopwords=STOPWORDS).generate(txt)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title('cluster ' + str(cluster))

    plt.show()


def search_word(word, data_train, frequency_threshold=2):
    results = dict()

    for row_idx in range(len(data_train)):
        # get tokens in the current row
        txt = ''
        txt += data_train.at[row_idx, TITLE_COLUMN]
        txt += '\n' + data_train.at[row_idx, KEYWORD_COLUMN]
        txt += '\n' + data_train.at[row_idx, ABSTRACT_COLUMN]
        txt = 'a ' + txt + ' a'
        txt = txt.replace('\n', ' ').replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('  ', ' ').lower()
        tokens = txt.split(' ')

        # find phrases containing the given word
        for token_idx in range(1, len(tokens) - 1): # ignore the added words
            current = tokens[token_idx]
            next = tokens[token_idx + 1]
            previous = tokens[token_idx - 1]

            if current == word:
                if previous not in STOPWORDS and next not in STOPWORDS:
                    result = previous + ' ' + current.upper() + ' ' + next
                    if result not in results:
                        results[result] = 1
                    else:
                        results[result] += 1

                if previous not in STOPWORDS:
                    result = previous + ' ' + current.upper()
                    if result not in results:
                        results[result] = 1
                    else:
                        results[result] += 1

                if next not in STOPWORDS:
                    result = current.upper() + ' ' + next.upper()
                    if result not in results:
                        results[result] = 1
                    else:
                        results[result] += 1
            else:
                continue

    sorted_results = sorted(results.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    # filter the most important words
    final_results = []

    for sorted_result in sorted_results:
        if sorted_result[1] >= frequency_threshold:
            final_results.append([sorted_result[0], sorted_result[1]])  # (word, frequency)
        else:
            break

    return final_results


if __name__ == '__main__':
    data_train = pd.read_csv('../data/[UCI] AAAI-14 Accepted Papers - Papers.csv')
    data_train = preprocess_data(data_train)
    X = data_train.drop(labels=['authors', 'groups', 'abstract', 'topics', 'keywords', 'title'], axis=1).to_numpy()

    choose_the_best_clustering(X)

    # after looking at the plot, I decided to choose the number of clusters = 10
    chosen_num_cluster = 10
    kmeans = KMeans(n_clusters=chosen_num_cluster, random_state=0).fit(X)
    clustering = kmeans.fit(X)

    # plot the data to see the clustering result
    plot_data_in_2d(X, clustering.labels_, chosen_num_cluster)

    # generate word cloud
    generate_word_cloud(kmeans.labels_, data_train, chosen_num_cluster)
    print('Searching the phrases containing the given word')
    final_results = search_word('learning', data_train)
    print('Results: ' + str(final_results))
