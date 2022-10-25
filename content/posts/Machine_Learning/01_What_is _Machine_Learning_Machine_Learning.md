---
title: "What is Machine Learning"
date: "2021-05-06"
tags: ["Machine Learning"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---

 > **Arthur Samuel (1959)**. *Machine Learning: The field of study that gives computers the ability to learn without being explicitly learned*.

 > **Tom Mitchell (1998)**. *Well-posed Learning Problem: a computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E*.

 <div align="center">
<img src="/img_ML/1_Five_Steps.PNG" width=650px />
 </div>

 <br>

## Representation Algorithms Grouped By Learning Style

There are different ways an algorithm can model a problem based on its interaction with the experience or environment or whatever we want to call the input data. It is popular in machine learning and artificial intelligence textbooks to first consider the learning styles that an algorithm can adopt.

This taxonomy or way of organizing machine learning algorithms is useful because it forces you to think about the roles of the input data and the model preparation process and select one that is the most appropriate for your problem in order to get the best result.

### Supervised Learning Algorithms（监督学习）

<div align="center">
<img src="/img_ML/1_Supervised_Learning.PNG" width=600px/>
</div>

<br>

In supervised learning, we are given a data set and **already know what our correct output** should look like, having the idea that there is a relationship between the input and the output.

Supervised learning problems are categorized into **"Regression"** and **"Classification"** problems. In a regression problem, we are trying to predict results within a continuous output, meaning that we are trying to map input variables to some continuous function. In a classification problem, we are instead trying to predict results in a discrete output. In other words, we are trying to map input variables into discrete categories.

**Examples of Supervised Learning:**  Regression, Decision Tree, Random Forest, KNN, Logistic Regression etc.

### Unsupervised Learning Algorithms （无监督学习）

<div align="center">
<img src="/img_ML/1_Unsupervised_Learning.PNG" width=600px/>
</div>

<br>

In unsupervised learning, the data has no labels. Unsupervised learning allows us to approach problems with **little or no idea** what our results should look like. We can derive structure from data where we don't necessarily know the effect of the variables. We can derive this structure by **Clustering** the data based on relationships among the variables in the data. With unsupervised learning there is no feedback based on the prediction results.

**Examples of Unsupervised Learning:**  Apriori algorithm, K-means.

### Reinforcement Learning Algorithms （强化学习）

<div align="center">
<img src="/img_ML/1_Reinforcement_Learning.PNG" width=600px/>
</div>

<br>

Lastly, we have reinforcement learning, the latest frontier of machine learning. A reinforcement algorithm learns by trial and error to achieve a clear objective. It tries out lots of different things and is **rewarded** or **penalized** depending on whether its behaviors help or hinder it from reaching its objective. This is like giving and withholding treats when teaching a dog a new trick. Reinforcement learning is the basis of Google’s AlphaGo, the program that famously beat the best human players in the complex game of Go.

**Example of Reinforcement Learning:** Markov Decision Process

## Representation Algorithms Grouped By Similarity

Algorithms are often grouped by similarity in terms of their function (how they work). For example, tree-based methods, and neural network inspired methods. This is a useful grouping method, but it is not perfect. There are still algorithms that could just as easily fit into multiple categories like Learning Vector Quantization that is both a neural network inspired method and an instance-based method. There are also categories that have the same name that describe the problem and the class of algorithm such as Regression and Clustering.

<br>

### Regression Algorithms（回归算法）

Regression is concerned with modeling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model. Regression methods are a workhorse of statistics and have been co-opted into statistical machine learning. This may be confusing because we can use regression to refer to the class of problem and the class of algorithm. Really, regression is a process.


The most popular regression algorithms are:

<div>
<img src="/img_ML/1_Regression.PNG" width=135px align='right'/>
</div>

* Ordinary Least Squares Regression (OLSR)
* **Linear Regression**
* **Logistic Regression**
* Stepwise Regression
* Multivariate Adaptive Regression Splines (MARS)
* Locally Estimated Scatterplot Smoothing (LOESS)

<br>

### Instance-based Algorithms（基于核的算法）

Instance-based learning model is a decision problem with instances or examples of training data that are deemed important or required to the model.

Such methods typically build up a database of example data and compare new data to the database using a similarity measure in order to find the best match and make a prediction. For this reason, instance-based methods are also called **winner-take-all methods** and **memory-based learning**. Focus is put on the representation of the stored instances and similarity measures used between instances.

The most popular instance-based algorithms are:

<div>
<img src="/img_ML/1_Instance_Base.PNG" width=135px align='right'/>
</div>

* **K-Nearest Neighbor (KNN)**
* Learning Vector Quantization (LVQ)
* Self-Organizing Map (SOM)
* Locally Weighted Learning (LWL)
* **Support Vector Machines (SVM)**

<br>

### Decision Tree Algorithms（决策树算法）

Decision tree methods construct a model of decisions made based on actual values of attributes in the data. Decisions fork in tree structures until a prediction decision is made for a given record. Decision trees are trained on data for classification and regression problems. Decision trees are often fast and accurate and a big favorite in machine learning.

The most popular regularization algorithms are:

<div>
<img src="/img_ML/1_Decision_Tree.PNG" width=135px align='right'/>
</div>

* Classification and Regression Tree (CART)
* Iterative Dichotomiser 3 (ID3)
* C4.5 and C5.0 (different versions of a powerful approach)
* Chi-squared Automatic Interaction Detection (CHAID)
* Decision Stump
* M5
* Conditional Decision Trees

<br>

### Regularization Algorithms（正则化算法）

An extension made to another method (typically regression methods) that penalizes models based on their complexity, favoring simpler models that are also better at generalizing. I have listed regularization algorithms separately here because they are popular, powerful and generally simple modifications made to other methods.

<div>
<img src="/img_ML/1_Regularzation.PNG" width=135px align='right'/>
</div>

The most popular regularization algorithms are:

* Ridge Regression
* Least Absolute Shrinkage and Selection Operator (LASSO)
* Elastic Net
* Least-Angle Regression (LARS)

<br>

### Bayesian Algorithms（贝叶斯算法）

Bayesian methods are those that explicitly apply Bayes’ Theorem for problems such as classification and regression.

The most popular Bayesian algorithms are:

<div>
<img src="/img_ML/1_Bayesian.PNG" width=135px align='right'/>
</div>

* **Naive Bayes**
* **Gaussian Naive Bayes**
* Multinomial Naive Bayes
* Averaged One-Dependence Estimators (AODE)
* Bayesian Belief Network (BBN)
* Bayesian Network (BN)

<br>

### Clustering Algorithms（聚类算法）

Clustering, like regression, describes the class of problem and the class of methods. Clustering methods are typically organized by the modeling approaches such as centroid-based and hierarchal. All methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

<div>
<img src="/img_ML/1_Clustering.PNG" width=135px align='right'/>
</div>

The most popular clustering algorithms are:

* **K-Means**
* K-Medians
* Expectation Maximisation (EM)
* **Hierarchical Clustering**

<br>

### Association Rule Learning Algorithms（关联规则学习算法）

Association rule learning methods extract rules that best explain observed relationships between variables in data. These rules can discover important and commercially useful associations in large multidimensional datasets that can be exploited by an organization.

<div>
<img src="/img_ML/1_Association_Rule.PNG" width=135px align='right'/>
</div>

The most popular association rule learning algorithms are:

* Apriori algorithm
* Eclat algorithm

<br>

### Dimensionality Reduction Algorithms（降维算法）

Like clustering methods, dimensionality reduction seek and exploit the inherent structure in the data, but in this case in an unsupervised manner or order to summarize or describe data using less information. This can be useful to visualize dimensional data or to simplify data which can then be used in a supervised learning method. Many of these methods can be adapted for use in classification and regression.

The most popular dimensionality reduction algorithms are:

<div>
<img src="/img_ML/1_Dimensionality_Reduction.PNG" width=135px align='right'/>
</div>

* **Principal Component Analysis (PCA)**
* Principal Component Regression (PCR)
* Partial Least Squares Regression (PLSR)
* Sammon Mapping
* Multidimensional Scaling (MDS)
* Projection Pursuit
* **Linear Discriminant Analysis (LDA)**
* Mixture Discriminant Analysis (MDA)
* **Quadratic Discriminant Analysis (QDA)**
* **Flexible Discriminant Analysis (FDA)**

<br>

### Artificial Neural Network Algorithms（人工神经网络算法）

Artificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks. They are a class of pattern matching that are commonly used for regression and classification problems but are really an enormous subfield comprised of hundreds of algorithms and variations for all manner of problem types.

Note that Deep Learning have been separated out from neural networks because of the massive growth and popularity in the field. Here we are concerned with the more classical methods.

The most popular artificial neural network algorithms are:

<div>
<img src="/img_ML/1_Artificial_Neural_Network.PNG" width=135px align='right'/>
</div>

* **Perceptron**
* Multilayer Perceptrons (MLP)
* **Back-Propagation**
* Stochastic Gradient Descent
* Hopfield Network
* **Radial Basis Function Network (RBFN)**

<br>

### Deep Learning Algorithms（深度学习算法）

Deep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation. They are concerned with building much larger and more complex neural networks and, as commented on above, many methods are concerned with very large datasets of labelled analog data, such as image, text. audio, and video.

The most popular deep learning algorithms are:

<div>
<img src="/img_ML/1_Deep_Learning.PNG" width=135px align='right'/>
</div>

* **Convolutional Neural Network (CNN)**
* **Recurrent Neural Networks (RNNs)**
* Long Short-Term Memory Networks (LSTMs)
* Stacked Auto-Encoders
* Deep Boltzmann Machine (DBM)
* Deep Belief Networks (DBN)

<br>

### Ensemble Algorithms（集成算法）

Ensemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction. Much effort is put into what types of weak learners to combine and the ways in which to combine them. This is a very powerful class of techniques and as such is very popular.

The most popular ensemble algorithms are:

<div>
<img src="/img_ML/1_Ensemble.PNG" width=135px align='right'/>
</div>

* **Boosting**
* Bootstrapped Aggregation (Bagging)
* **AdaBoost**
* Weighted Average (Blending)
* Stacked Generalization (Stacking)
* Gradient Boosting Machines (GBM)
* Gradient Boosted Regression Trees (GBRT)
* **Random Forest**

<br>

### Other Machine Learning Algorithms（其他机器学习算法）

Algorithms from specialty subfields of machine learning, such as:

* Computational intelligence (evolutionary algorithms, etc.)
* Computer Vision (CV)
* Natural Language Processing (NLP)
* Recommender Systems
* Reinforcement Learning
* Graphical Models
* And more…

<br>

##  Feature Selection Algorithms

When building a machine learning model in real-life, it’s almost rare that all the variables in the dataset are useful to build a model. Adding redundant variables reduces the generalization capability of the model and may also reduce the overall accuracy of a classifier. Furthermore adding more and more variables to a model increases the overall complexity of the model. The goal of feature selection in machine learning is to find the best set of features that allows one to build useful models of studied phenomena. The most popular feature selection algorithms are:

* **Chi-square Test**
* **Fisher’s Score**
* **Correlation Coefficient**
* Forward Feature Selection
* Backward Feature Elimination
* **L1 Regularization**

<br>

## Performance Measures

Evaluating the machine learning algorithm is an essential part of any project. Most of the times we use classification accuracy to measure the performance of our model, however it is not enough to truly judge our model. The metrics that you choose to evaluate your machine learning model is very important. Choice of metrics influences how the performance of machine learning algorithms is measured and compared. The most popular evaluation performance measures are:

* Classification Accuracy
* Logarithmic Loss
* Confusion Matrix
* Area under Curve
* F1 Score
* Mean Absolute Error
* Mean Squared Error

<br>

## Optimization Algorithms

Optimization is the problem of finding a set of inputs to an objective function that results in a maximum or minimum function evaluation. The most common type of optimization problems encountered in machine learning are continuous function optimization, where the input arguments to the function are real-valued numeric values, e.g. floating point values. The output from the function is also a real-valued evaluation of the input values. The most popular optimization algorithms are:

* Greedy Search
* Beam Search
* Gradient decent
* Conjugate gradient
* Momentum
* Adagrad
* RMSProp
* Adam

<br>

## References

[1]  Brownlee, J. (2020, August 14). A Tour of Machine Learning Algorithms. Machine Learning Mastery. https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/.

[2]  Contact Centric. (2021, March 26). Machine Learning: A Quick Introduction and Five Core Steps. Centric Consulting. https://centricconsulting.com/blog/machine-learning-a-quick-introduction-and-five-core-steps/.

[3]  Brownlee, J. (2020, August 20). How to Choose a Feature Selection Method For Machine Learning. Machine Learning Mastery. https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/.
