---
title: "Principal Component Analysis"
date: "2021-05-16"
tags: ["Machine Learning", "PCA"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---

Principal Component Analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

The goal is to preserve as much of the variance in the original data as possible in the new coordinate systems.
Give data on `$d$` variables, the hope is that the data points will lie mainly in a linear subspace of dimension lower than `$d$`. In practice, the data will usually not lie precisely in some lower dimensional subspace. The new variables that form a new coordinate system are called **principal components** (PCs).

> **主成分分析(principal components analysis,PCA)** 是一种简化数据集的技术。它是一个线性变换。这个变换把数据变换到一个新的坐标系统中，其中，第一个新坐标轴选择是原始数据中 **方差最大** 的方向，第二个新坐标轴选取是与 **第一个坐标轴正交的平面中** 使得方差最大的，第三个轴是与第 1,2 个轴正交的平面中方差最大的。依次类推，可以得到n个这样的坐标轴。通过这种方式获得的新的坐标轴，我们发现，大部分方差都包含在前面 `$k$`个坐标轴中，后面的坐标轴所含的方差几乎为 0。于是，我们可以忽略余下的坐标轴，只保留前面k个含有绝大部分方差的坐标轴. 这相当于只保留包含绝大部分方差的维度特征，而 **忽略包含方差几乎为 0 的特征维度** ，实现对数据特征的 **降维处理** 。

<br>

## Principal Component Analysis

* PCs are denoted by `$u_{1},u_{2},...,u_{d}.$`
* The principal components form a basis for the data.
* Since PCs are orthogonal linear transformations of the original variables there is at most `$d$` PCs.
* Normally, not all of the `$d$` PCs are used but rather a subset of `$p$` PCs, `$u_{1},u_{2},...,u_{p}.$`
* In order to approximate the space spanned by the original data points
`$x = \begin{bmatrix}
x_{1}    \\
\cdots   \\
x_{d}   \\
\end{bmatrix}_{\ dx1}$`
We can choose `$p$` based on what percentage of the variance of the original data we would like to maintain.

The first PC, `$u_{1}$` is called **first principal component** and has the maximum variance, thus it accounts for the most
significant variance in the data.

The second PC, `$u_{2}$` is called **second principal component** and has the second highest variance and so on until PC ud which has the minimum variance.


<div align="center">
<img src="/img_ML/5_PC12.PNG" width=400px/>
</div>

<br>

In order to capture as much of the variability as possible, let us choose the first principal component, denoted by
`$u_{1}$`, to capture the maximum variance. Suppose that all centred observations are stacked into the columns of a `$d \times n$` matrix
`$X = \begin{bmatrix}
x_{1} ... x_{n}  \\
\end{bmatrix}_{d \times n}$`,
where each column corresponds to a `$d$`-dimensional observation and there are `$n$` observations. The projection of `$n$`, `$d$`-dimensional observations on the first principal component `$u_{1}$` is `$u_{1}^{T}X$`. We want projection on this first dimension to have maximum variance.

`$$
\frac{1}{2N}\sum_{n=1}^{N}(u_{1}^{T}x_{n}-u_{1}^{T}\bar{x}_{n})^{2} = Var(u_{1}^{T}X)= u_{1}^{T}Su_{1} \\
$$`

Where `$S$` is the `$d \times d$` sample covariance matrix of `$X$`.


Clearly `$Var(u_{1}^{T}X)$` can be made arbitrarily large by increasing the magnitude of `$u_{1}$`. This means that the variance stated above has no upper limit and so we can not find the maximum. To solve this problem, we choose `$u_{1}$` to maximize `$u_{1}^{T}Su_{1}$` while constraining `$u_{1}$` to have unit length. Therefore, we can rewrite the above optimization problem as:

`$$
\mathrm{max} \ \ u_{1}^{T}Su_{1} \\
\mathrm{Subject \ \ to} \ \ u_{1}^{T}u_{1} = 1 \\
$$`

To solve this optimization problem a Lagrange multiplier `$\lambda$` is introduced:

`$$
L(u_{1}, \lambda) = u_{1}^{T}Su_{1}-\lambda(u_{1}^{T}u_{1}-1) \\
$$`

### Lagrange Multiplier for PCA
Lagrange multipliers are used to find the maximum or minimum of a function `$f (x, y)$` subject to constraint
`$g(x, y) = c$`. we define a new constant `$\lambda$` called a **Lagrange Multiplier** and we form the Lagrangian,

`$$
L(x,y,\lambda) = f(x,y)-\lambda g(x,y) \\
$$`

If `$f(x^{*},y^{*})$` is the max of `$f(x, y)$` , there exists `$\lambda^{*}$` such that `$(x^{*},y^{*},\lambda^{*})$` is a stationary point of `$L$` (partial derivatives are 0). In addition `$(x^{*},y^{*})$` is a point in which functions `$f$` and `$g$` touch but do not cross. At this point, the tangents of `$f$` and `$g$` are parallel or gradients of `$f$` and `$g$` are parallel, such that:

`$$
\nabla_{x,y}f = \lambda\nabla_{x,y}g \\
$$`

`$$
\begin{align*}
\mathrm{Where}, \ \nabla_{x,y}f &= (\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}) \mathrm{\to the \ gradient \ of} \ f \\
\nabla_{x,y}g &= (\frac{\partial g}{\partial x}, \frac{\partial g}{\partial y}) \mathrm{\to the \ gradient \ of} \ g \\
\end{align*}
$$`

Differentiating with respect to `$u_{1}$` gives `$d$` equations,

`$$
\frac{\partial L(u_{1}, \lambda)}{\partial u_{1}} = 2S u_{1} - 2\lambda u_{1} = 0 \\
S u_{1} = \lambda u_{1} \\
$$`

Premultiplying both sides by `$u_{1}^{T}$` we have:

`$$
u_{1}^{T}S u_{1} = \lambda u_{1}^{T} u_{1} = \lambda \\
$$`

`$ u_{1}^{T}S u_{1}$` is maximized if `$\lambda$` is the largest eigenvalue of `$S$`. Clearly `$\lambda$` and `$u_{1}$` are an eigenvalue and an eigenvector of `$S$`. Differentiating `$L(u_{1}, \lambda)$` with respect to the Lagrange multiplier `$\lambda$` gives us back the constraint:

`$$
u_{1}^{T}u_{1} = 1 \\
$$`

This shows that the first principal component is given by the eigenvector with the largest associated eigenvalue of the sample covariance matrix `$S$`. A similar argument can show that the `$p$` dominant eigenvectors of covariance matrix `$S$` determine the first `$p$` principal components.

Note that the PCs decompose the total variance in the data in the following way :

`$$
\sum_{i=1}^{d}Var(u_{i}^{T}X) = \sum_{i=1}^{d}u_{1}^{T}Su_{1} = \sum_{i=1}^{d}(\lambda_{i}) = Tr(S) = Var(X) \\
$$`

`$Var(u_{1}^{T}X)$` is maximized if `$u_{1}$` is the eigenvector of `$S$` with the corresponding maximum eigenvalue. Each successive PC can be generated in the above manner by taking the eigenvectors of `$S$` that correspond to the eigenvalues:

`$$
\lambda_{1} \geq ... \geq \lambda_{d} \\
$$`

Such that

`$$
Var(u_{1}^{T}X) \geq ... \geq Var(u_{d}^{T}X) \\
$$`

### Direct PCA Algorithm
* **Normalize the data:** Set `$X = X - \bar{X}$`
* **Recover basis (PCs):** Calculate `$XX^{T}=\sum_{i=1}^{n}x_{i}x_{i}^{T}$` and let `$U=$` eigenvectors of `$XX^{T}$` corresponding to the top `$p$` eigenvalues.
* **Encode training data:** `$Y=U^{T}X$` Where `$Y$` is a `$p \times n$` matrix of encodings of the original data.
* **Reconstruct training data:** `$\hat{X} = UY=UU^{T}X$`.
* **Encode test example:** `$y=U^{T}x$` where `$y$` is a `$p$`-dimensional encoding of x.
* **Reconstruct test example:** `$\hat{x} = Uy = UU^{T}x$`.

`$$
U^{T}_{p \times d} \cdot
(X=\begin{bmatrix}
x_{1}    \\
x_{2}    \\
\vdots        \\
x_{n}    \\
\end{bmatrix}_{\ d \times n})\longrightarrow
(Y=\begin{bmatrix}
y_{1}    \\
y_{2}    \\
\cdots        \\
y_{p}    \\
\end{bmatrix}_{\ p \times n}) \\
$$`

> **基于特征值分解(Eigendecomposition)协方差矩阵实现PCA算法** : 对数据去中心化后，计算协方差矩阵(Covariance Matrix) `$XX^{T}$`。用 **特征值分解方法** 求协方差矩阵的特征值(Eigenvalues)与特征向量(Eigenvectors)。对特征值从大到小排序，选择其中最大的 k 个。然后将其对应的 k 个特征向量分别作为行向量组成特征向量矩阵。最后将数据转换到 k 个特征向量构建的新空间中。

A unique solution can be obtained by finding the **singular value decomposition** of `$X$`. For each rank `$p$`, `$U$` consists of the first `$p$` columns of `$U$`.

`$$
X=U\Sigma V^{T} \\
$$`

**The columns of `$U$` in the SVD contain the eigenvectors of `$XX^{T}$`.**

> **基于SVD(Singular Value Decomposition)分解协方差矩阵实现PCA算法** : 对数据去中心化后，计算协方差矩阵(Covariance Matrix) `$XX^{T}$`。通过 **SVD** 计算协方差矩阵的特征值(Eigenvalues)与特征向量(Eigenvectors)。 对特征值从大到小排序，选择其中最大的 k 个。然后将其对应的 k 个特征向量分别作为行向量组成特征向量矩阵。最后将数据转换到 k 个特征向量构建的新空间中。

>当样本数多、样本特征数也多的时候,先求出协方差矩阵 `$XX^{T}$` 的计算是很大的。然而有一些SVD的实现算法可以先 **不求出协方差矩阵也能求出右奇异矩阵** 。也就是说，我们的 PCA 算法可以不用做特征分解而是通过 SVD 来完成，这个方法在样本量很大的时候很有效。实际上，scikit-learn的 PCA 算法的背后真正的实现就是用的 SVD，而不是特征值分解

<br>

## Referencce

[1] Raschka, S. (2014, April 13). Implementing a Principal Component Analysis (PCA). Dr. Sebastian Raschka. https://sebastianraschka.com/Articles/2014_pca_step_by_step.htmlprincipal-component-analysis-pca-vs-multiple-discriminant-analysis-mda.

[2] 张洋. (n.d.). 数据的向量表示及降维问题. CodingLabs. http://blog.codinglabs.org/articles/pca-tutorial.html.
