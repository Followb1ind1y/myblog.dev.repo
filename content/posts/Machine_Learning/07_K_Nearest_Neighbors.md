---
title: "K-Nearest Neighbors"
date: "2021-05-18"
tags: ["Machine Learning", "K-Nearest Neighbors"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---


K Nearest Neighbor(KNN) is a very simple, easy to understand, versatile and one of the topmost machine learning algorithms. KNN used in the variety of applications such as finance, healthcare, political science, handwriting detection, image recognition and video recognition. KNN algorithm used for both **classification** and **regression** problems.

We say that KNN is a non-parametric and lazy learning algorithm. **Non-parametric** means there is no assumption for underlying data distribution. In other words, the model structure determined from the dataset. This will be very helpful in practice where most of the real world datasets do not follow mathematical theoretical assumptions. **Lazy algorithm** means it does not need any training data points for model generation. All training data used in the testing phase. This makes training faster and testing phase slower and costlier. Costly testing phase means time and memory. In the worst case, KNN needs more time to scan all data points and scanning all data points will require more memory for storing training data.

<br>

> **KNN(K Nearest Neighbor)** 是通过测量不同特征值之间的距离进行分类。它的思路是：如果一个样本在特征空间中的 k 个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别，其中 K 通常是不大于 20 的整数。KNN算法中，所选择的邻居都是已经正确分类的对象。该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

## K-Nearest Neighbors

* Training example in Euclidean space: `$x\in\mathbf{R}^{d}$`

* **Idea:** The value of the target function for a new query is estimated from the known value(s) of the nearest training example(s)

* Distance typically defined to be Euclidean:

`$$
\parallel x^{(a)}-x^{(b)} \parallel_{2} = \sqrt{\sum_{j=1}^{d}(x_{j}^{(a)} - x_{j}^{(b)})^{2}} \\
$$`

### kNN Algorithom

1. Load the training and test data
2. Choose the value of K
3. For each point in test data:
* find the Euclidean distance to all training data points
* store the Euclidean distances in a list and sort it
* choose the first k points
* assign a class to the test point based on the majority of classes present in the chosen points
4. End

<div align="center">
<img src="/img_ML/7_knn_algorithom.PNG" width=800px/>
</div>

<br>

> KNN算法的思想大致为: 在训练集中数据和标签已知的情况下，输入测试数据，将测试数据的特征与训练集中对应的特征进行相互比较，找到训练集中与之最为相似的 **前 K 个数据** ，则该测试数据对应的类别就是 K 个数据中 **出现次数最多的那个分类** 。如下图，绿色圆要被决定赋予哪个类。如果 K=3，由于红色三角形所占比例为 2/3，绿色圆将被赋予红色三角形那个类，如果 K=5，由于蓝色四方形比例为 3/5，因此绿色圆被赋予蓝色四方形类。

<br>

<div align="center">
<img src="/img_ML/7_knn_cercle.PNG" width=300px/>
</div>

<br>

### Choice of K

* `$k=1$` can be unstable, particularly if the data are noisy
* Better to choose an odd number to avoid ties, e.g. `$k=3$` or `$k=5$`
* Larger `$k$` may lead to better performance. But if we set `$k$` too large we may end up looking at samples that are not neighbors (are far away from the query)
* Rule of thumb is `$k < \mathrm{Sqrt}(n)$`, where `$n$` is the number of training examples
* Choose `$k$` that yields the smallest error on the test data

<br>

## Pros and Cons of K-Nearest Neighbors

### Pros:
* It is extremely easy to implement
* It is lazy learning algorithm and therefore requires no training prior to making real time predictions. This makes the KNN algorithm much faster than other algorithms that require training e.g SVM, linear regression, etc.
* Since the algorithm requires no training before making predictions, new data can be added seamlessly.
* There are only two parameters required to implement KNN i.e. the value of K and the distance function (e.g. Euclidean or Manhattan etc.)

### Cons:
* Accuracy depends on the quality of the data.
* Poor at classifying data points in a boundary where they can be classified one way or another.
* Doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate distance in each dimension.
* Has a high prediction cost for large datasets. This is because in large datasets the cost of calculating distance between new point and each existing point becomes higher.
* Doesn't work well with categorical features since it is difficult to find the distance between dimensions with categorical features.

### Issues & Remedies

* If some attributes (coordinates of x) have **larger ranges**, they are treated as more important
    * Normalize scale
         - Simple option: Linearly scale the range of each feature to be, e.g., in range `$[0,1]$`
         - Linearly scale each dimension to have 0 mean and variance 1 (compute mean `$\mu$` and variance `$\sigma^{2}$` for an attribute `$x_{j}$` and scale: `$\frac{(x_{j} - m)}{\sigma}$`)
    * Be careful: sometimes scale matters

* **Irrelevant**, **correlated** attributes add noise to distance measure
    * eliminate some attributes
    * or vary and possibly adapt weight of attributes

* **Non-metric** attributes (symbols)
    * Hamming distance

* **Expensive at test time:** To find one nearest neighbor of a query point `$x$`, we must compute the distance to all `$N$` training examples. Complexity: `$O(kdN)$` for kNN
    * Use subset of dimensions
    * Pre-sort training examples into fast data structures (e.g., kd-trees)
    * Compute only an approximate distance (e.g., LSH)
    * Remove redundant data (e.g., condensing)

* **Storage Requirements:** Must store all training data
    * Remove redundant data (e.g., condensing)
    * Pre-sorting often increases the storage requirements

* **High Dimensional Data:** “Curse of Dimensionality”
    * Required amount of training data increases exponentially with dimension
    * Computational cost also increases

<br>

## Reference

[1] Zemel, R., Urtasun, R., &amp; Fidler, S. (n.d.). CSC 411: Lecture 05: Nearest Neighbors. https://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/05_nn.pdf.

[2] Sanjay.M. (2018, November 2). KNN using scikit-learn. Medium. https://towardsdatascience.com/knn-using-scikit-learn-c6bed765be75.
