---
title: "Neural Network: Radial Basis Function Neural Networks (RBN)"
date: "2021-07-04"
tags: ["RBN", "SURE"]
categories: ["Data Science", "Deep Learning", "Neural Network"]
weight: 3
---

In **Single Perceptron / Multi-layer Perceptron(MLP)**, we only have linear separability because they are composed of input and output layers(some hidden layers in MLP). We at least need one hidden layer to derive a non-linearity separation. Our **RBN** what it does is, it transforms the input signal into another form, which can be then feed into the network to get **linear separability**. RBN is structurally same as perceptron(MLP).

<div align="center">
  <img src="/img_DL/02_MLP_RBN.PNG" width=400px/>
</div>

RBNN is composed of **input**, **hidden**, and **output** layer. RBNN is **strictly limited** to have exactly **one** hidden layer. We call this hidden layer as **feature vector**. We apply **non-linear transfer function** to the feature vector before we go for classification problem. When we increase the dimension of the feature vector, the linear separability of feature vector increases.

## Network Structure

<div align="center">
  <img src="/img_DL/02_RBN_Structure.PNG" width=350px/>
</div>

### 1. Input :

`$$
\{(x_{i},y_{i})\}_{i=1}^{n} \ , \ \mathrm{where} \ x_{i} \subset \mathbb{R}^{d} \\
$$`

`$$
X =
\begin{bmatrix}
x_{11} \ \cdots \ x_{1n} \\
\vdots \ \ddots \ \vdots \\
x_{d1} \ \cdots \ x_{dn} \\
\end{bmatrix}_{\ d\times n} =
\begin{bmatrix}
\vdots \ \ \ \vdots \\
x_{1} \ \cdots \  x_{n} \\
\vdots \ \ \ \vdots \\
\end{bmatrix}_{\ d\times n} \ , \
Y =
\begin{bmatrix}
y_{11} \ \cdots \ y_{1n} \\
\vdots \ \ddots \ \vdots \\
y_{k1} \ \cdots \ y_{kn} \\
\end{bmatrix}_{\ k\times n}
$$`

### 2. Radial Basis Function

* We define a `$\mathrm{receptor} = t$`.
* We draw confrontal maps around the `$\mathrm{receptor}$`.
* Gaussian Functions are generally used for Radian Basis Function(confrontal mapping). So we define the radial distance `$r = \parallel x- t \parallel.$`
* There are many choices for the basis function. The commonly used is:

`$$
\phi_{j}(x_{i}) = e^{-|x_{i}\ - \ \mu_{j}|^{2}}
$$`

<div align="center">
  <img src="/img_DL/02_Radial_Function.jpeg" width=350px/>
</div>

### 3. Output:

`$$
y_{k}(x)=\sum_{j=1}^{m}W_{jk}\phi_{j}(x)
$$`

`$$
W =
\begin{bmatrix}
w_{11} \ \cdots \ w_{1k} \\
\vdots \ \ddots \ \vdots \\
w_{m1} \ \cdots \ w_{mk} \\
\end{bmatrix}_{\ m\times k} \ , \
\phi =
\begin{bmatrix}
\phi_{1}(x_{1}) \ \cdots \ \phi_{1}(x_{n}) \\
\vdots \ \ddots \ \vdots \\
\phi_{m}(x_{1}) \ \cdots \ \phi_{m}(x_{n}) \\
\end{bmatrix}_{\ m\times n}
$$`

The output will be:

`$$
Y = W^{T}\phi
$$`

where `$Y$` and `$\phi$` are known while `$W$` is unknown.

## Optimization

`$$
\psi = \parallel Y - W^{T}\phi \ \parallel^{2}
$$`

`$W$` can be computed by minimizing our objective function w.r.t `$w$`.

`$$
\min_{W}\parallel Y - W^{T}\phi \ \parallel^{2}
$$`

This optimization problem can be solved in close form:

`$$
\begin{align*}
\frac{\partial}{\partial W}\parallel Y - W^{T}\phi \ \parallel^{2} &= \frac{\partial}{\partial W}Tr[(Y - W^{T}\phi)^{T}(Y - W^{T}\phi)] \\
&= \frac{\partial}{\partial W}Tr[Y^{T}Y+\phi^{T}WW^{T}\phi-Y^{T}W^{T}\phi - \phi^{T}WY] \\
&= 0 + 2\phi\phi^{T}W - 2\phi Y^{T} = 0 \\
&\Rightarrow \phi\phi^{T}W = \phi Y^{T} \\
&\Rightarrow W = (\phi\phi^{T})^{-1}\phi Y^{T}
\end{align*}
$$`

In RBF network the estimated function is:

`$$
\begin{align*}
\hat{Y} &= W^{T}\phi \\
&= ((\phi\phi^{T})^{-1}\phi Y^{T})^{T}\phi \\
&= Y\phi^{T}((\phi\phi^{T})^{-1})^{T}\phi \\
\Rightarrow \hat{Y}^{T} &= \underbrace{\phi^{T}(\phi\phi^{T})^{-1}\phi}_{H} Y^{T} \\
\Rightarrow \hat{Y}^{T} &= HY^{T}
\end{align*}
$$`

## Stein’s unbiased risk estimator (SURE)

Assume `$
T = \{(x_{i},y_{i})\}_{i=1}^{n}
$` be the training set.

* `$f(\cdot)$` `$\to$` True model
* `$\hat{f}(\cdot)$` `$\to$` Estimated model
* err `$\to$` Empirical error: `$\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}$`
* Err `$\to$` True error: `$\frac{1}{n}\sum_{i=1}^{n}(\hat{f}_{i}-f_{i})^{2}$`
* `$y$` `$\to$` Observations

Also assume

`$$
y_{i} = f(x_{i}) + \varepsilon_{i} \ , \ \ \ \ \mathrm{Where} \ \varepsilon_{i} \sim N(0,\sigma^{2})
$$`

For point `$(x_{0},y_{0})$` we are interested in

`$$
\begin{align*}
E[(\hat{y}_{0}-y_{0})^{2}] &= E[(\hat{f}_{0}-f_{0}-\varepsilon_{0})^{2}] \\
&= E[((\hat{f}_{0}-f_{0})-\varepsilon_{0})^{2}] \\
&= E[(\hat{f}_{0}-f_{0})^{2}+\varepsilon_{0}^{2}-2\varepsilon_{0}(\hat{f}_{0}-f_{0})] \\
&= E[(\hat{f}_{0}-f_{0})^{2}]+E[\varepsilon_{0}^{2}]-2E[\varepsilon_{0}(\hat{f}_{0}-f_{0})] \\
&= E[(\hat{f}_{0}-f_{0})^{2}]+\sigma^{2}-2E[\varepsilon_{0}(\hat{f}_{0}-f_{0})] \\
\end{align*}
$$`

### Case 1

Assume `$(x_{0},y_{0})\notin T$`.

In this case, since `$\hat{f}$` is estimated only based on points in training set, therefore it is completely independent from `$(x_{0},y_{0})$`

`$$
E[\varepsilon_{0}(\hat{f}_{0}-f_{0})] = 0 \\
\Rightarrow E[(\hat{y}_{0}-y_{0})^{2}] = E[(\hat{f}_{0}-f_{0})^{2}]+\sigma^{2}
$$`

If summing up all `$m$` points that are not in `$T$`.

`$$
\underbrace{\sum_{i=1}^{m}(\hat{y}_{i}-y_{i})^{2}}_{\mathrm{err}}= \underbrace{\sum_{i=1}^{m}(\hat{f}_{i}-f_{i})^{2}}_{\mathrm{Err}}+ m\sigma^{2} \\
\mathrm{err} = \mathrm{Err} + m\sigma^{2} \\
\mathrm{Err} = \mathrm{err} - m\sigma^{2}
$$`

Empirical error (`$\mathrm{err}$`) is a good estimator of true error (`$\mathrm{Err}$`) if the point `$(x_{0},y_{0})$` is not in the training set.

### Case 2

Assume `$(x_{0},y_{0})\in T$`. Then `$E[\varepsilon_{0}(\hat{f}_{0}-f_{0})] \neq 0$`

**Stein's Lemma**: If `$x \sim N(\theta,\sigma^{2})$` and `$g(x)$` differentiable. Then,

`$$
    E[g(x)(x-\theta)]=\sigma^{2}E[\frac{\partial g(x)}{\partial x}]
$$`

`$$
\begin{align*}
E[\underbrace{\varepsilon_{0}}_{(x-\theta)}\underbrace{(\hat{f}_{0}-f_{0})}_{g(x)}] &=\sigma^{2}E[\frac{\partial (\hat{f}_{0}-f_{0})}{\partial \varepsilon_{0}}] \\
&= \sigma^{2}E[\frac{\partial \hat{f}_{0}}{\partial \varepsilon_{0}}-\underbrace{\frac{\partial f_{0}}{\partial \varepsilon_{0}}}_{0}] \\
&= \sigma^{2}E[\frac{\partial \hat{f}_{0}}{\partial \varepsilon_{0}}] \\
&= \sigma^{2}E[\frac{\partial \hat{f}_{0}}{\partial y_{0}}\cdot\underbrace{\frac{\partial y_{0}}{\partial \varepsilon_{0}}}_{1}] \\
&= \sigma^{2}E[\frac{\partial \hat{f}_{0}}{\partial y_{0}}] \\
&= \sigma^{2}E[D_{0}]
\end{align*}
$$`

Thus,

`$$
E[(\hat{y}_{0}-y_{0})^{2}] = E[(\hat{f}_{0}-f_{0})^{2}]+\sigma^{2} - 2\sigma^{2}E[D_{0}]
$$`

Sum over all `$n$` data points:

`$$
\underbrace{\sum_{i=1}^{n}(\hat{y}_{i}-y_{i})^{2}}_{\mathrm{err}}= \underbrace{\sum_{i=1}^{n}(\hat{f}_{i}-f_{i})^{2}}_{\mathrm{Err}}+ n\sigma^{2} -2\sigma^{2}\sum_{i=1}^{n}D_{i} \\
\mathrm{Err} = \mathrm{err} - n\sigma^{2}+\underbrace{2\sigma^{2}\sum_{i=1}^{n}D_{i}}_{\mathrm{Complexity \ of \ model}} \Rightarrow \mathrm{Stein’s \ Unbiased \ Risk \ Estimator \ (SURE)}
$$`

## Coplexity control for RBN

Let's apply **SURE** to **RBF**:

`$$
 \left.
    \begin{array} \\
        D_{i} = \frac{\partial \hat{f}_{i}}{\partial y_{i}} \\
        \hat{f}_{i} = \hat{y}_{i} = H_{i:y}
    \end{array}
\right \}D_{i}=\frac{\partial \hat{f}_{i}}{\partial y_{i}} = \frac{\partial H_{i:y}}{\partial y_{i}} = H_{ii}
$$`

Then **SURE** will be:

`$$
\begin{align*}
\mathrm{Err} &= \mathrm{err} - n\sigma^{2}+2\sigma^{2}\sum_{i=1}^{n}H_{ii} \\
&= \mathrm{err} - n\sigma^{2}+2\sigma^{2}Tr(H) \\
&= \mathrm{err} - n\sigma^{2}+2\sigma^{2}Tr[\phi^{T}(\phi\phi^{T})^{-1}\phi] \\
&= \mathrm{err} - n\sigma^{2}+2\sigma^{2}Tr[\underbrace{\phi\phi^{T}(\phi\phi^{T})^{-1}}_{I\to m\times m}] \\
&= \mathrm{err} - n\sigma^{2}+2\sigma^{2}Tr[I_{m}] \\
&= \mathrm{err} - n\sigma^{2}+2\sigma^{2}m \\
\end{align*}
$$`

For computing SURE, we need to know the value of `$\sigma$`. But we do not know it. Therefore we need to estimate it.

`$$
\sigma^{2} = \frac{\sum_{i=1}^{n}(y_{i}-\hat{y}_{i})^{2}}{n-1}
$$`

is the function of complexity (more complex, smaller `$\sigma^{2}$`), in practice, we do not consider the `$\hat{y}$` to be the function of complexity and instead we consider it to be a low bias and high variance estimation (for example a line). With this assumption the `$\sigma$` will be considered to be constant and independent from the complexity of model.

## Reference

[1] McCormick, C. (2013, August 15). Radial Basis Function Network (RBFN) Tutorial. Radial Basis Function Network (RBFN) Tutorial · Chris McCormick. https://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/.
