---
title: "Boosting"
date: "2021-05-30"
tags: ["Machine Learning", "Boosting"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---


**Boosting** is an ensemble modeling technique which attempts to build a strong classifier from the number of weak classifiers. It is done by building a model using weak models in series. First, a model is built from the training data. Then the second model is built which tries to correct the errors present in the first model. This procedure is continued and models are added until either the complete training data set is predicted correctly or the maximum number of models are added.

Boosting being a sequential process, each subsequent model attempts to correct the errors of the previous model. It is focused on reducing the bias unlike bagging. It makes the boosting algorithms prone to overfitting. To avoid overfitting, parameter tuning plays an important role in boosting algorithms.

<div align="center">
<img src="/img_ML/12_Boosting.PNG" width=600px/>
</div>

<br>

A **Weak Classifier** is one whose error rate is only slightly better than random guessing. Theoretically a weak classifier can be boosted to perform pretty well. To find weak learners, we apply base learning (ML) algorithms with a different distribution. As each time base learning algorithm is applied, it generates a new weak prediction rule. This is an iterative process. After many iterations, the boosting algorithm combines these weak rules into a single strong prediction rule.

Boosting combines many weak classifiers to produce one powerful "committee". The weak classifiers do not have equal weight. For classification into two categories labeled `$\{-1,1\}$`,

`$$
f(x) = \mathrm{sign}[\sum_{j=1}^{m}\alpha_{j}h_{j}(x)] \\
$$`

where `$h_{j}(x)$` is a weak learner and `$\alpha_{j}$` are weights.

<br>

> **Boosting** 和 bagging 最本质的差别在于他对基础模型不是一致对待的，而是经过不停的考验和筛选来挑选出「精英」，然后给精英更多的投票权，表现不好的基础模型则给较少的投票权，然后综合所有人的投票得到最终结果。大部分情况下，经过 boosting 得到的结果偏差(bias)更小。

## AdaBoost (for classification)

Adaptive Boosting, or most commonly known **AdaBoost**, is a Boosting algorithm. This algorithm uses the method to correct its predecessor. It pays more attention to under fitted training instances by the previous model. Thus, at every new predictor the focus is more on the complicated cases more than the others.

It fits a sequence of weak learners on different weighted training data. It starts by predicting the original data set and gives equal weight to each observation. If prediction is incorrect using the first learner, then it gives higher weight to observation which have been predicted incorrectly. Being an iterative process, it continues to add learner(s) until a limit is reached in the number of models or accuracy.

### AdaBoost Algorithm

`$$
\{(x_{i},y_{i})\}_{i=1}^{n}, \ \ \ \ \ \ \ \ \ \ y_{i}=\{-1,1\}, \ \ \ \ \ \ \ \ \ \ f(x) = \mathrm{sign}[\sum_{j=1}^{m}\alpha_{j}h_{j}(x)]
$$`

1. Initialize weights `$w_{i}=\frac{1}{n}, \  i = 1,2,..,n$`


2. For`$\ j=1$` to `$m$`:

      * a) Fit a classifier `$h_{j}(x)$` to the training data
      * b) Compute `$err_{j}=L_{j}=\frac{\sum_{i=1}^{n}w_{i}I(y_{i}\neq h(x_{i}))}{\sum_{i=1}^{n}w_{i}}$`
      * c) Compute `$\alpha_{j}=log(\frac{1-L_{j}}{L_{j}})$`
      * d) Set `$w_{i} := w_{i}\exp[\alpha_{j}I(y_{i}\neq h(x_{j}))]$`


3. Final classification

`$$
h(x_{i}) =\mathrm{sign}[\sum_{j=1}^{m}\alpha_{j}h_{j}(x)]
$$`

*  If classified correctly, the weight of an observation remains unchanged.
*  If classified incorrectly, the weight is increased by multiplying
`$\exp(\alpha_{j})$`
* alpha varies with the degree of misclassification

<br>

##  Additive Model

Generally, boosting fits an additive model

`$$
f(x)=\sum_{j=1}^{m}\beta_{j}\phi_{j}(x)
$$`

Where `$\phi_{j}(x)$` are basis functions (weak learners).

Adaboost is restricted to **2‐class** classification, **boosting is not**. To fit the model a loss function has to be minimized:

`$$
\min_{\beta_{j},\gamma_{j}}L(y,f(x))=\sum_{i=1}^{n}L(y_{i},\sum_{j=1}^{m}\beta_{j}\phi_{j}(x_{i},\gamma_{j}))
$$`

Finding optimal coefficients for all `$m$` iterations simultaneously is difficult. We therefore simplify the problem. At each iteration, we find the best fit to the “residual” from the previous iteration.

* What "best" means depends on the loss function
* Definition of "residual" depends on the loss function

Sequential fit: values from earlier iterations are never changed.

###  Forward Stagewise Additive Modeling Algorithm

1. Initial `$f_{0}(x)$`


2. For `$j=1$` to `$m$`. Add to the existing model such that the loss function is minimized.
    * a) Minimize the loss `$\sum_{i=1}^{n}L[f_{j-1}(x_{i})+\beta_{j}\phi_{j}(x_{i},\gamma_{j})]$`
    * b) Update the function `$f_{j}(x)=f_{j-1}(x)+\beta_{j}\phi_{j}(x,\gamma_{j})$`


One can show that Adaboost is a forward stagewise additive model using an exponential loss function

`$$
\mathrm{Loss}(y,h(x)) = \exp(-y\times h(x))
$$`

<br>

## Gradient Boosting

Gradient Boosting is another very popular Boosting algorithm which works pretty similar to what we’ve seen for AdaBoost. Gradient Boosting works by sequentially adding the previous predictors underfitted predictions to the ensemble, ensuring the errors made previously are corrected.

The difference lies in what it does with the underfitted values of its predecessor. Contrary to AdaBoost, which tweaks the instance weights at every interaction, this method tries to fit the new predictor to the residual errors made by the previous predictor.

### Gradient Boosting Algorithm:

1. A model is built on a subset of data.
2. Using this model, predictions are made on the whole dataset.
3. Errors are calculated by comparing the predictions and actual values.
4. A new model is created using the errors calculated as target variable. Our objective is to find the best split to minimize the error.
5. The predictions made by this new model are combined with the predictions of the previous.
6. New errors are calculated using this predicted value and actual value.
7. This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.

<br>

## XGBoost

XG Boost or Extreme Gradient Boosting is an advanced implementation of the Gradient Boosting. This algorithm has high predictive power and is ten times faster than any other gradient boosting techniques. Moreover, it includes a variety of regularization which reduces overfitting and improves overall performance.

### Advantages of XGBoost

* It implements regularization which helps in reducing overfit (Gradient Boosting does not have);
* It implements parallel processing which is much faster than Gradient Boosting;
* Allows users to define custom optimization objectives and evaluation criteria adding a whole new dimension to the model;
* XGBoost has an in-built routine to handle missing values;
* XGBoost makes splits up to the max_depth specified and then starts pruning the tree backwards and removes splits beyond which there is no positive gain;
* XGBoost allows a user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.

<br>

## Reference

[1]  Sarkar, P., S, A., &amp; Shah, P. (2019, October 22). What is Boosting and AdaBoost in Machine Learning? Knowledgehut. https://www.knowledgehut.com/blog/data-science/boosting-and-adaboost-in-machine-learning.

[2]  Navlani, A. (2018, November 18). AdaBoost Classifier in Python. DataCamp Community. https://www.datacamp.com/community/tutorials/adaboost-classifier-python.
