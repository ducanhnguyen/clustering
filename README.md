# clustering
k-means, soft k-means, etc.

### Example 1

Apply k-means to cluster the digit-recognizer dataset.

Because we actually have the true ground set, we can evaluate the quality of k-means using purity.


### Example 2

The raw implementation of soft-kmeans

Soft-kmeans solves partially the sensitivity of initialization of k-means.


### Example 3

<b>Step 1.</b> Choose the best number of clusters (called K).

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_num_cluster.png" width="650">

 Based on the trend of inertia line and silhouette line, I choose K = 10

<b>Step 2.</b> Plot the dataset on 2D with K = 10

Use PCA or tSNE to transform high dimensional dataset into 2d dataset

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_2d.png" width="650">

<b>Step 3.</b> Plot the word cloud of 10 clusters

<img src="https://github.com/ducanhnguyen/clustering/blob/master/img/wordcloud_.png" width="650">

<b>Step 4.</b> Search the phrases containing a word of a word cloud

With the word 'learning':
Result: ['machine LEARNING', 35], ['transfer LEARNING', 27], ['reinforcement LEARNING', 26], ['LEARNING ALGORITHMS', 24], ['metric LEARNING', 19], ['online LEARNING', 14], ['feature LEARNING', 14], ['active LEARNING', 14], ['semi-supervised LEARNING', 11], ['deep LEARNING', 11], ['dictionary LEARNING', 10], ['manifold LEARNING', 9], ['LEARNING ALGORITHM', 9], ['representation LEARNING', 8], ['multi-task LEARNING', 8], ['structure LEARNING', 7], ['multi-label LEARNING', 7], ['machine LEARNING algorithms', 7], ['LEARNING METHODS', 7], ['LEARNING APPROACH', 7], etc.
