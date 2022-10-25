---
title: "Fisher’s Linear Discriminant Analysis"
date: "2021-05-17"
tags: ["Machine Learning", "FDA"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---


**Fisher's Linear Discriminant Analysis (FDA)** is most commonly used as dimensionality reduction technique in the pre-processing step for pattern-classification and machine learning applications. The goal is to project a dataset onto a lower-dimensional space with good class-separability in order avoid overfitting ("curse of dimensionality") and also reduce computational costs. The general FDA approach is very similar to a Principal Component Analysis, but in addition to finding the component axes that maximize the variance of our data (PCA), we are additionally interested in the axes that maximize the separation between multiple classes.

<br>

## Fisher’s Linear Discriminant Analysis

Assume we have only 2 classes. The idea behind Fisher’s Linear Discriminant Analysis is to reduce the dimensionality of the data to one dimension. That is, to take d-dimensional `$x\in \mathbf{R}^{d}$` and map it to one dimension by finding `$w^{T}x$` where:

`$$z = w^{T}x=
\begin{bmatrix}
w_{1} ... w_{d}   \\
\end{bmatrix}
\begin{bmatrix}
x_{1}    \\
...      \\
x_{d}    \\
\end{bmatrix} = \sum_{i=1}^{d}w_{i}x_{i} \\
$$`

The one-dimensional `$z$` is then used for classification.

**Goal:** To find a direction such that projected data `$w^{T}x$` are well separated.

Consider the two-class problem:

`$$
\mu_{0}=\frac{1}{n_{0}}\sum_{i:y_{i}=0}x_{i} \ \ \ \ \ \ \ \ \ \ \ \ \mu_{1}=\frac{1}{n_{1}}\sum_{i:y_{i}=0}x_{i} \\
$$`

We want to:

1. **Maximize the distance between projected class means.**

2. **Minimize the within class variance.**

> **Fisher判别分析(Fisher’s Linear Discriminant Analysis)的目的**：给定一个投影向量`$w$`，将`$x$`投影到`$w$`向量上，使得不同类别的`$x$`投影后的值 `$y=w^{T}x$` 尽可能互相分开(far apart)。投影向量的选择很关键，当投影向量选择不合理时，不同类别的`$x$`投影后的`$y$`根本无法被分开。所以要找到最优的投影向量，使得投影后的值被最大限度地区分。(这里的前提是：原始数据是线性可分的)
>
>最大限度的划分投影后的值需要用到两个准则:
> 1. 投影后的两类样本 **均值** 之间的距离 **尽可能大**
> 2. 投影后两类样本各自的 **方差尽可能小**

<div align="center">
<img src="/img_ML/6_PCA_goal.jpeg" width=500px/>
</div>

<br>

The distance between projected class means is:

`$$
\begin{align*}
(w^{T}\mu_{0} - w^{T}\mu_{1})^{2} &= (w^{T}\mu_{0} - w^{T}\mu_{1})^{T}(w^{T}\mu_{0} - w^{T}\mu_{1}) \\
&= (\mu_{0}-\mu_{1})^{T}ww^{T}(\mu_{0}-\mu_{1}) \\
&=w^{T}(\mu_{0}-\mu_{1})(\mu_{0}-\mu_{1})^{T}w \\
&= w^{T}S_{B}w \\
\end{align*}
$$`
where `$S_{B}$` is the between-class variance (known).

Minimizing the within-class variance is equivalent to minimizing the sum of all individual within-class variances.
Thus the within class variance is:

`$$
\begin{align*}
w^{T}\Sigma_{0}w+w^{T}\Sigma_{1}w &= w^{T}(\Sigma_{0}+\Sigma_{1})w \\
&= w^{T}S_{W}w \\
\end{align*}
$$`

where `$S_{W}$` is the within-class covariance (known).

To maximize the distance between projected class means and minimize the within-class variance, we can maximize the ratio:

`$$
\max_{w} \ \frac{w^{T}S_{B}w}{w^{T}S_{W}w} \\
$$`

Note that the numerator is unbounded since we can make any arbitary `$w^{T}$`. But since we are only interested in direction, length is not important. Therefore we can fix the length of `$w$` (i.e. unit length) and find the direction. This is equivalent to finding:

`$$
\max_{w} \ w^{T}S_{B}w \\
\mathrm{Subject \  to} \ \ w^{T}S_{W}w = 1 \\
$$`

To turn this constraint optimization problem into a non-constranst optimization problem, we apply **Lagrange multipliers:**

`$$
L(w,\lambda) = w^{T}S_{B}w - \lambda(w^{T}S_{W}w-1) \\
$$`

Differentiating with respect to `$w$` we get:

`$$
\frac{\partial L}{\partial w} = 2S_{B} w - \lambda2S_{W} w = 0 \\
S_{B} w = \lambda S_{W}w \\
$$`

This is a generalized eigenvector problem that is equivalent to (if `$S_{W}$` is not singular):

`$$
S_{W}^{-1}S_{B}w = \lambda w \\
$$`

where `$\lambda$` and `$w$` are the eigenvalues and eigenvectors of `$S_{W}^{-1}S_{B}$`respectively. `$w$` is the eigenvector corresponding to the largest eigenvalue of `$S_{W}^{-1}S_{B}$`.

In fact, for two-classes problems, there exists a **simpler solution**. Recall that `$S_{B}w = (\mu_{0}-\mu_{1})(\mu_{0}-\mu_{1})^{T}w$` where `$(\mu_{0}-\mu_{1})^{T}w$` is a scalar. Therefore `$S_{B}w\propto(\mu_{0}-\mu_{1})$`. That is, `$S_{B}w$` is on the same direction as `$(\mu_{0}-\mu_{1})$`. Since `$S_{W}^{-1}S_{B}w = \lambda w$`, we get:

`$$
S_{w}^{-1}(\mu_{0}-\mu_{1}) \propto w \\
$$`

which gives us the direction.

<br>

## Fisher’s Linear Discriminant Analysis For Multiple Classes

<div align="center">
<img src="/img_ML/6_Multi_LDA.jpeg" width=600px/>
</div>

<br>

We have defined `$\varepsilon(w)=\frac{w^{T}S_{B}w}{w^{T}S_{W}w}$` that needs to be maximized. `$w$` is the largest eigen vectors of `$S_{W}^{-1}S_{B}$`. For two classes, `$w \propto S_{w}^{-1}(\mu_{0}-\mu_{1})$`. For `$k$`-class problem, Fisher Discriminant Analysis involves `$(k - 1)$` discriminant functions. Make `$W_{d \times (K-1)}$` where each column describes a discriminant. So now, we have to update the two notions we have defined for a `$2$`-class problem, `$S_{B}$` and `$S_{W}$` .

`$$
S_{w}=\sum_{i=1}^{K}\Sigma_{i} \\
$$`

`$S_{B}$` generalization to multiple classes in not obvious. We will define the total variance `$S_{T}$` as the sum of the within class variance and between classes variance.

`$$
S_{T} = S_{B}+S_{W} \\
$$`

Where `$S_{T}=\frac{1}{n}\sum_{i=1}^{n}(x_{i}-\mu)(x_{i}-\mu)^{T}$` and `$\mu=\frac{1}{n}\sum_{i=1}^{n}x_{i}$`. So,

`$$
S_{B} = S_{T}-S_{W} \\
$$`

It can be shown that `$W$` is the first `$(k - 1)$` eigen vectors of `$S_{W}^{-1}S_{B}$`.

<br>

## PCA vs. LDA

Both Linear Discriminant Analysis (LDA) and Principal Component Analysis (PCA) are linear transformation techniques that are commonly used for dimensionality reduction. PCA can be described as an “unsupervised” algorithm, since it “ignores” class labels and its goal is to find the directions (the so-called principal components) that maximize the variance in a dataset. In contrast to PCA, LDA is “supervised” and computes the directions (“linear discriminants”) that will represent the axes that that maximize the separation between multiple classes.

Although it might sound intuitive that LDA is superior to PCA for a multi-class classification task where the class labels are known, this might not always the case.

<div align="center">
<img src="/img_ML/6_PCA_vs_LDA.PNG" width=800px/>
</div>

<br>

> **PCA** 实质上是在寻找一个子空间。而这个子空间是协方差矩阵的特征空间(特征向量对应的空间)，选取特征值最大的 k 个特征向量组成的特征子空间(相当于这个子空间有 k 维，每一维代表一个特征，这 ｋ 个特征基本可以涵盖 90% 以上的信息)。**Fisher判别** 和 PCA 是在做类似的一件事，都是在找子空间。不同的是, PCA 是找一个低维的子空间，样本投影在这个空间基本不丢失信息。而 Fisher是寻找这样的一个空间，样本投影在这个空间上，类内距离最小，类间距离最大。
>
> 两者的 **相同点** ：
> * 两者均可以对数据进行降维。
> * 两者在降维时均使用了矩阵特征分解的思想。
> * 两者都假设数据符合高斯分布。
>
> 两者的 **不同点** ：
> * LDA是有监督的降维方法，而PCA是无监督的降维方法
> * LDA降维最多降到类别数k-1的维数，而PCA没有这个限制。
> * LDA除了可以用于降维，还可以用于分类。
> * LDA选择分类性能最好的投影方向，而PCA选择样本点投影具有最大方差的方向。

## Reference

[1] Raschka, S. (2014, August 3). Linear Discriminant Analysis. Dr. Sebastian Raschka. https://sebastianraschka.com/Articles/2014_python_lda.html.
