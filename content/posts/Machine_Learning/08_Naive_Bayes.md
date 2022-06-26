---
title: "Naive Bayes"
date: "2021-05-19"
tags: ["Machine Learning", "Naive Bayes"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---

**Naive Bayes classifiers** are linear classifiers that are known for being simple yet very efficient. The probabilistic model of naive Bayes classifiers is based on **Bayes' theorem**, and the adjective naive comes from the assumption that the features in a dataset are mutually independent. In practice, the independence assumption is often violated, but naive Bayes classifiers still tend to perform very well under this unrealistic assumption. Especially for small sample sizes, naive Bayes classifiers can outperform the more powerful alternatives.

Being relatively robust, easy to implement, fast, and accurate, naive Bayes classifiers are used in many different fields. Some examples include the diagnosis of diseases and making decisions about treatment processes, the classification of RNA sequences in taxonomic studies, and spam filtering in e-mail clients.

However, **strong violations of the independence assumptions** and **non-linear classification problems** can lead to very poor performances of naive Bayes classifiers. We have to keep in mind that the type of data and the type problem to be solved dictate which classification model we want to choose. In practice, it is always recommended to compare different classification models on the particular dataset and consider the prediction performances as well as computational efficiency.

<br>

> **朴素贝叶斯(Naive Bayes)** 的思想基础是：对于给出的待分类项，求解在此项出现的条件下各个类别出现的概率，哪个最大，就认为此待分类项属于哪个类别。**朴素** : 之所以称为朴素，是因为它假定数据集中的所有要素都是相互独立的。**贝叶斯** : 它基于贝叶斯定理。

## Naive Bayes

From Bayes Rule, we have:

`$$
P(Y=k|X=x) = \frac{p(X=x|Y=k)P(Y=k)}{P(X=x)} = \frac{\pi_{k}f_{k}(x)}{\sum_{l=1}^{K}\pi_{l}f_{l}(x)} \\
$$`

Where density is `$f_{k}(X) = P(X=x|Y=k)$` and prior probability of class `$k$` is `$\pi_{k} = P(Y=k)$`.

Conditional on class `$k$`, assume the variables `$x_{j}$` are independent:

`$$
f_{k}(x)= \prod_{j=1}^{p}f_{kj}(x_{j}) \\
$$`

Where p is the number of x‐variables.

* Independence assumption: Conditional on the outcome, there is no multicollinearity
* This assumption is almost always wrong; but extremely convenient.
* This method is also called "Idiot's Bayes"

Plugging the density into Bayes rule, we obtain:

`$$
P(Y=k|X=x) = \frac{\pi_{k}f_{k}(x)}{\sum_{l=1}^{K}\pi_{l}f_{l}(x)}=\frac{\pi_{k}\prod_{j=1}^{p}f_{kj}(x_{j})}{\sum_{l=1}^{K}\pi_{l}\prod_{j=1}^{p}f_{lj}(x_{j})} \\
$$`

The denominator does not depend on class `$k$`. It is a constant. To find the class that maximizes the posterior probability, we can ignore the denominator:

`$$
P(Y=k|X=x) \propto \pi_{k}\prod_{j=1}^{p}f_{kj}(x_{j}) \\
$$`

where `$j=1,...,p$` indexes x‐variables

Predict the class `$k$` that maximizes the posterior probability(Decision rule):

`$$
h(x) = \arg\max_{k}\left(P(Y=k)\prod_{j=1}^{p}P(X_{j}=x_{j}|Y=k)\right) = \arg\max_{k}\left(\pi_{k}\prod_{j=1}^{p}f_{kj}(x_{j})\right) \\
$$`

When there are many x‐variables, multiplying many small probabilities may result in an "underflow". Numerically, all posterior probabilities are 0. It is unclear which 0 is "largest". We can take the log to avoid this problem. Because it is a monotone function, taking the log does not change which class `$k$` gives the maximum posterior probability:

`$$
h(x) = \arg\max_{k}(log(\pi_{k}) + \sum_{j=1}^{p}log[f_{kj}(x_{j})]) \\
$$`

Typically, estimate the prior probability as the fraction of time the class occurs in the training data:

`$$
\pi_{k} = P(Y=k) = \frac{n_{k}}{n} \\
$$`

Estimate the probability as:

`$$
f_{kj}(x_{j}) = P(X_{j}=x_{j}|Y=k) = \frac{n_{kj}}{n_{k}} \\
$$`

   * where `$n_{k}$` is the number of obs in class `$k$`
   * Where `$n_{kj}$` is the number of obs in class `$k$` taking the value `$x_{j}$`

### LaPlace Smoothing
For a given x‐variable, **LaPlace smoothing** adds one observation to each x‐category

`$$
f_{kj}(x_{j}) = P(X_{j}=x_{j}|Y=k) = \frac{n_{kj}+1}{n_{k}+d_{j}} \\
$$`

* where `$d_{j}$` is the number of categories of the corresponding x‐variable

 ### Smoothing in general

* The probability estimates “shrink” away from the extremes
* Instead of adding just one observation, we can add an arbitrary number of observations, `$L$`, that controls the amount of shrinking:

`$$
f_{kj}(x_{j}) = P(X_{j}=x_{j}|Y=k) = \frac{n_{kj}+L}{n_{k}+L*d_{j}} \\
$$`

* In the limit, for very large L,

`$$
P(X_{j}=x_{j}|Y=k) \to \frac{L}{L \times d_{j}} = \frac{1}{d_{j}} \\
$$`

<div align="center">
<img src="/img_ML/8_LaPlace.PNG" width=800px/>
</div>

<br>

## Gaussian Naive Bayes

Gaussian Naive Bayes classifier assumes that the likelihoods are Gaussian:

`$$
f_{kj}(x_{j}) = P(X_{j}=x_{j}|Y=k) =\frac{1}{\sqrt{2\pi\sigma_{jk}^{2}}}exp[\frac{-(x_{j}-\mu_{jk})^{2}}{2\sigma_{jk}}] \\
$$`

* Gaussian Naive Bayes is not as common as the case where all x‐ variables are categorical.
* Maximum likelihood estimate of parameters are:

`$$
\begin{align*}
\mu_{jk} &= \frac{\sum_{n=1}^{N}I[Y^{(n)}=k]\cdot x_{j}^{(n)}}{\sum_{n=1}^{N}I[Y^{(n)}=k]} \\
\sigma_{jk} &= \frac{\sum_{n=1}^{N}I[Y^{(n)}=k]\cdot (x_{j}^{(n)}-\mu_{jk})^{2}}{\sum_{n=1}^{N}I[Y^{(n)}=k]} \\
\end{align*}
$$`

Let's consider a simple example using the Iris data set:

* There are 3 class labels: Setosa, Versicolor, Virginica which we label as `$y\in\{0,1,2\}$`
* There are two explanatory variables (features): `$X_{1}$`: sepal length and `$X_{2}$`: sepal width.

For each feature, we calculate the estimated class mean, class variance and prior probability
* **Mean:** `$\mu_{x_{1}|Y=0}$`, `$\mu_{x_{1}|Y=1}$`, `$\mu_{x_{1}|Y=2}$` and `$\mu_{x_{2}|Y=0}$`, `$\mu_{x_{2}|Y=1}$`, `$\mu_{x_{2}|Y=2}$`
* **Variance:** `$\sigma_{x_{1}|Y=0}^{2}$`, `$\sigma_{x_{1}|Y=1}^{2}$`, `$\sigma_{x_{1}|Y=2}^{2}$` and `$\sigma_{x_{2}|Y=0}^{2}$`, `$\sigma_{x_{2}|Y=1}^{2}$`, `$\sigma_{x_{2}|Y=2}^{2}$`
* **Prior:** `$P(Y=0)$`, `$P(Y=1)$`, `$P(Y=2)$`

For any point `$(x1,x2)$` we compute the Gaussian Naive Bayes objective function (i.e. the one we are trying to maximize) for each class :

`$$
\begin{align*}
h(x) &= \arg\max_{k}[P(Y=k)\prod_{j=1}^{2}P(X_{j}=x_{j}|Y=k)] \\
&= \arg\max_{k}[P(X_{1}=x_{1}|Y=k)P(k)\cdot P(X_{2}=x_{2}|Y=k)P(k)] \\
&= \arg\max_{k}[\phi(x_{1}|\mu_{x_{1}|k},\sigma_{x_{1}|k}^{2})P(k)\cdot \phi(x_{2}|\mu_{x_{2}|k},\sigma_{x_{2}|k}^{2})P(k)] \\
\end{align*}
$$`

* where `$\phi(x_{1}|\mu_{x_{1}|k},\sigma_{x_{1}|k}^{2})$` is the PDF of a Gaussian univariate distribution with parameters `$\mu_{x_{1}|k}$`, `$\sigma_{x_{1}|k}^{2}$`. Repeat this calculation for each class, and then predict the class which has the highest value.

<br>

## Reference

[1] Sicotte, X. B. (2018, June 22). Xavier Bourret Sicotte. Gaussian Naive Bayes Classifier: Iris data set - Data Blog. https://xavierbourretsicotte.github.io/Naive_Bayes_Classifier.html.
