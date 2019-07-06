# clustering
k-means, soft k-means, etc.

### Example 1

Apply k-means to cluster the minist dataset.

Dataset: https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer

Because we actually have the true ground set, we can evaluate the quality of k-means using purity.


### Example 2

The raw implementation of soft-kmeans

Dataset is generated automatically by using blob.

Soft-kmeans solves partially the sensitivity of initialization of k-means.

The following figure shows the result of clustering over iterations. The initial centroids are initialized randomly which are any data point in the dataset. These centroids would be updated over iterations until the convergence occurs.

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/blod_iterations.png" width="950">

### Example 3

<b>Step 1.</b> Choose the best number of clusters (called K).

I only use column <i>groups</i> to find the best model of clusters. The information in the column <i>groups</i> shows the trend of research.

I ignore the column <i>abstract</i> because this column is too detail, which is not showed the clear trend. Similarly, column <i>title</i> should be ignored with the same reason.

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_num_cluster.png" width="650">

Based on the trend of inertia line and silhouette line, I choose K = 10 where occurs elbow.

<b>Step 2.</b> Plot the dataset on 2D with K = 10

Use PCA or tSNE to transform high dimensional dataset into 2d dataset. There are 10 clusters denoted from 0 to 9.

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_2d.png" width="650">

<b>Step 3.</b> Plot the word cloud of 10 clusters

I collected the extracted information in the column 'title', 'keywords', and 'abstract' to create word cloud.

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_.png" width="850">

<b>Step 4.</b> Search the phrases containing a word of a word cloud

For example, with the word <i>learning'</i>:

Result: ['machine LEARNING', 35], ['transfer LEARNING', 27], ['reinforcement LEARNING', 26], ['LEARNING ALGORITHMS', 24], ['metric LEARNING', 19], ['online LEARNING', 14], ['feature LEARNING', 14], ['active LEARNING', 14], ['semi-supervised LEARNING', 11], ['deep LEARNING', 11], ['dictionary LEARNING', 10], ['manifold LEARNING', 9], ['LEARNING ALGORITHM', 9], ['representation LEARNING', 8], ['multi-task LEARNING', 8], ['structure LEARNING', 7], ['multi-label LEARNING', 7], ['machine LEARNING algorithms', 7], ['LEARNING METHODS', 7], ['LEARNING APPROACH', 7], etc.
