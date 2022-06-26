---
title: "[ML Basics] Probability and Information Theory"
date: "2021-06-24"
tags: ["Probability", "Information Theory"]
categories: ["Machine Learning Basics", "Data Science"]
weight: 3
---

## Random Variables

A **random variable** is a variable that can take on different values randomly. We typically denote the random variable itself with a lower case letter in plain typeface, and the values it can take on with lower case script letters. For example, `$x_{1}$` and `$x_{2}$` are both possible values that the random variable `$x$` can take on. For vector-valued variables, we would write the random variable as `$\mathrm{x}$` and one of its values as `$x$`. On its own, a random variable is just a description of the states that are possible; it must be coupled with a probability distribution that specifies how likely each of these states are.

Random variables may be **discrete** or **continuous**. A discrete random variable is one that has a finite or countably infinite number of states. Note that these states are not necessarily the integers; they can also just be named states that are not considered to have any numerical value. A continuous random variable is associated with a real value.

> **随机变量(random variable)** 是可以随机地取不同值的变量. 随机变量可以是离散的或者连续的. 离散随机变量拥有有限或者可数无限多的状态, 连续随机变量伴随着实数值.

## Probability Distributions

A **probability distribution** is a description of how likely a random variable or set of random variables is to take on each of its possible states. The way we describe probability distributions depends on whether the variables are discrete or continuous.

> **概率分布(probability distribution)** 用来描述随机变量或一簇随机变量在每一个可能取到的状态的可能性大小. 我们描述概率分布的方式取决于随机变量是离散的还是连续的.

### Discrete Variables and Probability Mass Functions

A probability distribution over discrete variables may be described using a **probability mass function (PMF)**. We typically denote probability mass functions with a capital `$P$`. Often we associate each random variable with a different probability mass function and the reader must infer which probability mass function to use based on the identity of the random variable, rather than the name of the function; `$P(x)$` is usually not the same as `$P(y)$`.

The probability mass function maps from a state of a random variable to the probability of that random variable taking on that state. The probability that `$\mathrm{x}=x$` is denoted as `$P(x)$`, with a probability of 1 indicating that `$\mathrm{x}=x$` is certain and a probability of 0 indicating that `$\mathrm{x}=x$` is impossible. Sometimes to disambiguate which PMF to use, we write the name of the random variable explicitly: `$P(\mathrm{x}=x)$`. Sometimes we define a variable first, then use `$\sim$` notation to specify which distribution it follows later: `$\mathrm{x} \sim P(\mathrm{x})$`.

> 离散型变量的概率分布可以用**概率质量函数(probability mass function, PMF)** 来描述. 概率质量函数将随机变量能够取得的每个状态映射到随机变量取得该状态的概率. 通常 `$\mathrm{x}=x$` 的概率用 `$P(x)$` 来表示.

Probability mass functions can act on many variables at the same time. Such a probability distribution over many variables is known as a **joint probability distribution**. `$P(\mathrm{x}=x,$` `$\mathrm{y}=y)$` denotes the probability that `$\mathrm{x}=x$` and `$\mathrm{y}=y$` simultaneously. We may also write `$P(x,y)$` for brevity.

> 概率质量函数可以同时作用于多个随机变量。这种多个变量的概率分布被称为**联合概率分布(joint probability distribution)**. `$P(\mathrm{x}=x,$` `$\mathrm{y}=y)$` 表示 `$\mathrm{x}=x$` 和 `$\mathrm{y}=y$` 同时发生的概率。我们也可以简写为 `$P(x,y)$`.

To be a probability mass function on a random variable `$x$`, a function `$P$` must satisfy the following properties:

* The domain of `$P$` must be the set of all possible states of `$\mathrm{x}$`.
* `$\forall x \in \mathrm{x}, 0 \leq P(x) \leq 1$`. An impossible event has probability 0 and no state can be less probable than that. Likewise, an event that is guaranteed to happen has probability 1, and no state can have a greater chance of occurring.
* `$\sum_{\forall x \in \mathrm{x}}P(x)=1$`. We refer to this property as being **normalized**. Without this property, we could obtain probabilities greater than one by computing the probability of one of many events occurring.

> 如果一个函数 `$P$` 是随机变量 `$x$` 的概率质量函数, 必须满足下面这几个条件:
> * `$P$` 的定义域必须是 `$x$` 所有可能状态的集合.
> * `$\forall x \in \mathrm{x}, 0 \leq P(x) \leq 1$`.
> * `$\sum_{\forall x \in \mathrm{x}}P(x)=1$`. 我们把这条性质称之为**归一化的(normalized)** .

For example, consider a single discrete random variable `$\mathrm{x}=x$` with `$k$` different states. We can place a **uniform distribution** on `$\mathrm{x}=x$`, that is, make each of its states equally likely—by setting its probability mass function to

`$$
P(\mathrm{x}=x_{i})= \frac{1}{k} \\
$$`

for all `$i$`. We can see that this fits the requirements for a probability mass function. The value `$\frac{1}{k}$` is positive because `$k$` is a positive integer. We also see that

`$$
\sum_{i}P(\mathrm{x}=x_{i})= \frac{1}{k}=\sum_{i}\frac{1}{k}=\frac{k}{k}=1 \\
$$`

so the distribution is properly normalized.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_PMF.PNG" width=400px/>
</div>
<br>

### Continuous Variables and Probability Density Functions

When working with continuous random variables, we describe probability distributions using a **probability density function (PDF)** rather than a probability mass function. To be a probability density function, a function `$p$` must satisfy the following properties:

* The domain of `$P$` must be the set of all possible states of `$\mathrm{x}$`.
* `$\forall x \in \mathrm{x}, P(x) \geq 0$`. Note that we do not require `$P(x) \leq 1$`.
* `$\int p(x)dx=1$`

> 当我们研究的对象是连续型随机变量时, 我们用**概率密度函数(probability density function, PDF)** 而不是概率质量函数来描述它的概率分布. 如果一个函数 `$p$` 是概率密度函数，必须满足下面这几个条件:
> *  `$p$` 的定义域必须是 `$x$` 所有可能状态的集合.
> *  `$\forall x \in \mathrm{x}, P(x) \geq 0$`
> *  `$\int_{\forall x \in \mathrm{x}} p(x)dx=1$`

A probability density function `$p(x)$` does not give the probability of a specific state directly, instead the probability of landing inside an infinitesimal region with volume `$\delta x$` is given by `$p(x)\delta x$`.

We can integrate the density function to find the actual probability mass of a set of points. Specifically, the probability that `$x$` lies in some set `$\mathbb{S}$` is given by the integral of `$p(x)$` over that set. In the univariate example, the probability that `$x$` lies in the interval `$[a,b]$` is given by `$\int_{[a,b]} p(x)dx$`.

For an example of a probability density function corresponding to a specific probability density over a continuous random variable, consider a uniform distribution on an interval of the real numbers. We can do this with a function `$u(x;a,b)$`, where `$a$` and `$b$` are the endpoints of the interval, with `$b>a$`. The ";" notation means "parametrized by" ; we consider `$x$` to be the argument of the function, while `$a$` and `$b$` are parameters that define the function. To ensure that there is no probability mass outside the interval, we say `$u(x;a,b)=0$` for all `$x \notin [a,b]$`. Within `$[a,b]$`, `$u(x;a,b)=\frac{1}{b-a}$`. We can see that this is nonnegative everywhere. Additionally, it integrates to 1. We often denote that `$xd$` follows the uniform distribution on `$[a,b]$` by writing `$x \sim U(a,b)$`.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_PDF.PNG" width=400px/>
</div>
<br>

## Marginal Probability


Sometimes we know the probability distribution over a set of variables and we want to know the probability distribution over just a subset of them. The probability distribution over the subset is known as the **marginal probability distribution**.

> 有时候，我们知道了一组变量的联合概率分布, 但想要了解其中一个子集的概率分布. 这种定义在子集上的概率分布被称为 **边缘概率分布(marginal probability distribution)** .

For example, suppose we have **discrete random variables** `$\mathrm{x}$` and `$\mathrm{y}$`, and we know `$P(\mathrm{x},\mathrm{y})$`. We can find `$P(\mathrm{x})$` with the **sum rule**:

`$$
\forall x \in \mathrm{x}, P(\mathrm{x}=x) = \sum_{y} P(\mathrm{x}=x,\mathrm{y}=y) \\
$$`

> 假设有离散型随机变量 `$\mathrm{x}$` 和 `$\mathrm{y}$`, 并且我们知道 `$P(\mathrm{x},\mathrm{y})$`. 我们可以依据下面的**求和法则(sum rule)** 来计算 `$P(\mathrm{x})$`:
> `$$
> \forall x \in \mathrm{x}, P(\mathrm{x}=x) = \sum_{y} P(\mathrm{x}=x,\mathrm{y}=y) \\
> $$`

The name “marginal probability” comes from the process of computing marginal probabilities on paper. When the values of `$P(\mathrm{x},\mathrm{y})$` are written in a grid with different values of `$x$` in rows and different values of `$y$` in columns, it is natural to sum across a row of the grid, then write `$P(x)$` in the margin of the paper just to the right of the row.

For **continuous variables**, we need to use integration instead of summation:

`$$
p(x) = \int p(x,y)dy \\
$$`

> 对于连续型变量，我们需要用积分替代求和:
> `$$
> p(x) = \int p(x,y)dy \\
> $$`

## Conditional Probability

In many cases, we are interested in the probability of some event, given that some other event has happened. This is called a **conditional probability**. We denote the conditional probability that `$\mathrm{y}=y$` given `$\mathrm{x}=x$` as `$P(\mathrm{y}=y|\mathrm{x}=x)$`. This conditional probability can be computed with the formula

`$$
P(\mathrm{y}=y|\mathrm{x}=x) = \frac{P(\mathrm{y}=y,\mathrm{x}=x)}{P(\mathrm{x}=x)} \\
$$`

The conditional probability is only defined when `$P(\mathrm{x}=x)>0$`. We cannot compute the conditional probability conditioned on an event that never happens.

> 在很多情况下, 我们感兴趣的是某个事件, 在给定其他事件发生时出现的概率. 这种概率叫做**条件概率(conditional probability)** . 我们将给定 `$\mathrm{x}=x$`, `$\mathrm{y}=y$` 发生的条件概率记为 `$P(\mathrm{y}=y|\mathrm{x}=x)$`. 这个条件概率可以通过下面的公式计算:
> `$$
> P(\mathrm{y}=y|\mathrm{x}=x) = \frac{P(\mathrm{y}=y,\mathrm{x}=x)}{P(\mathrm{x}=x)} \\
> $$`
> 条件概率只在 `$P(\mathrm{x}=x)>0$` 时有定义.

It is important not to confuse conditional probability with computing what would happen if some action were undertaken. The conditional probability that a person is from Germany given that they speak German is quite high, but if a randomly selected person is taught to speak German, their country of origin does not change. Computing the consequences of an action is called making an **intervention query**. Intervention queries are the domain of **causal modeling**.

## The Chain Rule of Conditional Probabilities

Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable:

`$$
P(\mathrm{x}^{(1)},\cdots,\mathrm{x}^{(n)})=P(\mathrm{x}^{(1)}\prod_{i=2}^{n}P(\mathrm{x}^{(i)}|\mathrm{x}^{(1)},\cdots,\mathrm{x}^{(i-1)}) \\
$$`

This observation is known as the chain rule or product rule of probability. It follows immediately from the definition of conditional probability. For example, applying the definition twice, we get

`$$
\begin{align*}
P(a,b,c) &= P(a|b,c)P(b,c) \\
P(b,c) &= P(b|c)P(c) \\
P(a,b,c) &= P(a|b,c)P(b|c)P(c) \\
\end{align*}
$$`

## Independence and Conditional Independence

Two random variables `$x$` and `$y$` are **independent** if their probability distribution can be expressed as a product of two factors, one involving only `$x$` and one involving only `$y$`:

`$$
\forall x \in \mathrm{x}, y \in \mathrm{y}, p(\mathrm{x}=x, \mathrm{y}=y) = p(\mathrm{x}=x)p(\mathrm{y}=y) \\
$$`

> 两个随机变量 `$x$` 和 `$y$`, 如果它们的概率分布可以表示成两个因子的乘积形式, 并且一个因子只包含 `$x$` 另一个因子只包含 `$y$`, 我们就称这两个随机变量是**相互独立的(independent)**

Two random variables `$x$` and `$y$` are **conditionally independent** given a random variable `$z$` if the conditional probability distribution over `$x$` and `$y$` factorizes in this way for every value of `$z$`:

`$$
\forall x \in \mathrm{x}, y \in \mathrm{y},  z \in \mathrm{z}, p(\mathrm{x}=x, \mathrm{y}=y | \mathrm{z}=z) = p(\mathrm{x}=x| \mathrm{z}=z)p(\mathrm{y}=y| \mathrm{z}=z) \\
$$`

> 如果关于 `$x$` 和 `$y$` 的条件概率分布对于 `$z$` 的每一个值都可以写成乘积的形式, 那么这两个随机变量 `$x$` 和 `$y$` 在给定随机变量 `$z$` 时是**条件独立的(conditionally independent)**.

We can denote independence and conditional independence with compact notation: `$\mathrm{x} \bot \mathrm{y}$` means that `$\mathrm{x}$` and `$\mathrm{y}$` are independent, while `$\mathrm{x} \bot \mathrm{y} | \mathrm{z}$` means that `$\mathrm{x}$` and `$\mathrm{y}$` are conditionally independent given `$\mathrm{z}$`.

## Expectation, Variance and Covariance

The expectation or expected value of some function `$f(x)$` with respect to a probability distribution `$P(x)$` is the average or mean value that `$f$` takes on when `$x$` is drawn from `$P$` . For discrete variables this can be computed with a summation:

`$$
E_{\mathrm{x} \sim P}[f(x)] = \sum_{x}P(x)f(x) \\
$$`

> 函数 `$f(x)$` 关于某分布 `$P(x)$` 的**期望(expectation)** 或者**期望值(expected value)** 是指，当 `$x$` 由 `$P$` 产生，`$f$` 作用于 `$x$` 时，`$f(x)$` 的平均值. 对于离散型随机变量，这可以通过求和得到:
> `$$ E_{\mathrm{x} \sim P}[f(x)] = \sum_{x}P(x)f(x) \\ $$`


while for continuous variables, it is computed with an integral:

`$$
E_{\mathrm{x} \sim P}[f(x)] = \int p(x)f(x)dx \\
$$`

> 对于连续型随机变量可以通过求积分得到:
> `$$ E_{\mathrm{x} \sim P}[f(x)] = \int p(x)f(x)dx \\ $$`

When the identity of the distribution is clear from the context, we may simply write the name of the random variable that the expectation is over, as in `$E_{\mathrm{x}}[f(x)]$`. If it is clear which random variable the expectation is over, we may omit the subscript entirely, as in `$E[f(x)]$`. By default, we can assume that `$E[\cdot]$` averages over the values of all the random variables inside the brackets. Likewise, when there is no ambiguity, we may omit the square brackets.

Expectations are linear, for example,

`$$
E_{\mathrm{x}}[\alpha f(x) + \beta g(x)] = \alpha E_{\mathrm{x}}[f(x)] + \beta E_{\mathrm{x}}[g(x)] \\
$$`

when `$\alpha$` and `$\beta$` are not dependent on `$x$`.

> 期望是线性, 例如
> `$$ E_{\mathrm{x}}[\alpha f(x) + \beta g(x)] = \alpha E_{\mathrm{x}}[f(x)] + \beta E_{\mathrm{x}}[g(x)] \\ $$`
> 其中 `$\alpha$` 和 `$\beta$` 不依赖于 `$x$` .

The **variance** gives a measure of how much the values of a function of a random variable x vary as we sample different values of `$\mathrm{x}$` from its probability distribution:

`$$
Var(f(x)) = E[(f(x)-E[f(x)])^{2}] \\
$$`

When the variance is low, the values of `$f(x)$` cluster near their expected value. The square root of the variance is known as the **standard deviation**.

> **方差(variance)** 衡量的是当我们对 `$x$` 依据它的概率分布进行采样时, 随机变量 `$x$` 的函数值会呈现多大的差异:
> `$$ Var(f(x)) = E[(f(x)-E[f(x)])^{2}] \\ $$`
> 当方差很小时, `$f(x)$` 的值形成的簇比较接近它们的期望值. 方差的平方根被称为 **标准差(standard deviation)** .

The **covariance** gives some sense of how much two values are linearly related to each other, as well as the scale of these variables:

`$$
Cov(f(x),g(y)) = E[(f(x)-E[f(x)])(g(y)-E[g(y)])] \\
$$`

> **协方差(covariance)** 在某种意义上给出了两个变量线性相关性的强度以及这些变量的尺度:
> `$$ Cov(f(x),g(y)) = E[(f(x)-E[f(x)])(g(y)-E[g(y)])] \\ $$`

High absolute values of the covariance mean that the values change very much and are both far from their respective means at the same time. If the sign of the covariance is positive, then both variables tend to take on relatively high values simultaneously. If the sign of the covariance is negative, then one variable tends to take on a relatively high value at the times that the other takes on a relatively low value and vice versa. Other measures such as **correlation** normalize the contribution of each variable in order to measure only how much the variables are related, rather than also being affected by the scale of the separate variables.

> 协方差的绝对值如果很大则意味着变量值变化很大并且它们同时距离各自的均值很远. 如果协方差是正的, 那么两个变量都倾向于同时取得相对较大的值. 如果协方差是负的, 那么其中一个变量倾向于取得相对较大的值的同时, 另一个变量倾向于取得相对较小的值, 反之亦然. 其他的衡量指标如**相关系数(correlation)** 将每个变量的贡献归一化, 为了只衡量变量的相关性而不受各个变量尺度大小的影响.

The **covariance matrix** of a random vector `$x \in \mathbb{R}^{n}$` is an `$n \times n$` matrix, such that

`$$
Cov(\mathrm{x})_{i,j} = Cov(\mathrm{x}_{i},\mathrm{x}_{j}) \\
$$`

The diagonal elements of the covariance give the variance:

`$$
Cov(\mathrm{x}_{i},\mathrm{x}_{i}) = Var(\mathrm{x}_{i}) \\
$$`

## Common Probability Distributions

Several simple probability distributions are useful in many contexts in machine learning.

### Bernoulli Distribution

The **Bernoulli distribution** is a distribution over a single binary random variable. It is controlled by a single parameter `$\phi \in [0,1]$`, which gives the probability of the random variable being equal to 1. It has the following properties:

`$$
P(\mathrm{x}=1) = \phi \\
P(\mathrm{x}=0) = 1-\phi \\
P(\mathrm{x}=x) =\phi^{x}(1-\phi)^{1-x} \\
E_{\mathrm{x}}[\mathrm{x}] = \phi \\
Var_{\mathrm{x}}(\mathrm{x}) = \phi(1-\phi) \\
$$`

### Gaussian Distribution

The most commonly used distribution over real numbers is the **normal distribution**, also known as the **Gaussian distribution**:

`$$
\mathcal{N}(x;\mu, \sigma^{2}) = \sqrt{\frac{1}{2\pi \sigma^{2}}}exp \left(-\frac{1}{2\sigma^{2}}(x-\mu)^{2} \right) \\
$$`

The two parameters `$\mu \in \mathbb{R}$` and `$\sigma \in (0, \infty)$` control the normal distribution. The parameter `$\mu$` gives the coordinate of the central peak. This is also the mean of the distribution: `$E[x]=\mu$`. The standard deviation of the distribution is given by `$\sigma$`, and the variance by `$\sigma^{2}$`.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Standard_Normal.PNG" width=600px/>
</div>
<br>

Normal distributions are a sensible choice for many applications. In the absence of prior knowledge about what form a distribution over the real numbers should take, the normal distribution is a good default choice for two major reasons.

First, many distributions we wish to model are truly close to being normal distributions. The **central limit theorem** shows that the sum of many independent random variables is approximately normally distributed. This means that in practice, many complicated systems can be modeled successfully as normally distributed noise, even if the system can be decomposed into parts with more structured behavior.

Second, out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty over the real numbers. We can thus think of the normal distribution as being the one that inserts the least amount of prior knowledge into a model.

The normal distribution generalizes to `$\mathbb{R}^{n}$`, in which case it is known as the **multivariate normal distribution**. It may be parametrized with a positive definite symmetric matrix `$\Sigma$`:

`$$
\mathcal{N}(x;\mu, \Sigma) = \sqrt{\frac{1}{(2\pi)^{n} det(\Sigma)}}exp\left(-\frac{1}{2}(x-\mu)^{T} \Sigma^{-1} (x-\mu) \right) \\
$$`

The parameter `$\mu$` still gives the mean of the distribution, though now it is vector-valued. The parameter `$\Sigma$` gives the covariance matrix of the distribution.

### Exponential and Laplace distributions

In the context of deep learning, we often want to have a probability distribution with a sharp point at `$x=0$`. To accomplish this, we can use the **exponential distribution**:

`$$
p(x;\lambda)=\begin{cases}
\begin{align*}
&\lambda e^{-\lambda x}  &x \geq 0 \\
&0  &x < 0 \\
\end{align*}
\end{cases}
$$`

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Exponential.PNG" width=500px/>
</div>
<br>

A closely related probability distribution that allows us to place a sharp peak of probability mass at an arbitrary point `$\mu$` is the **Laplace distribution**

`$$
\mathrm{Laplace}(x;\mu, \gamma) = \frac{1}{2\gamma}exp{\left(-\frac{|x-\mu|}{\gamma}\right)} \\
$$`

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Laplace.SVG" width=500px/>
</div>

## Useful Properties of Common Functions

Certain functions arise often while working with probability distributions, especially the probability distributions used in deep learning models.

One of these functions is the **logistic sigmoid**:

`$$
\sigma(x) = \frac{1}{1+exp(-x)} \\
$$`

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Logistic_Sigmoid.PNG" width=600px/>
</div>
<br>

The logistic sigmoid is commonly used to produce the `$\phi$` parameter of a Bernoulli distribution because its range is `$(0,1)$`, which lies within the valid range of values for the `$\phi$` parameter. The sigmoid function saturates when its argument is very positive or very negative, meaning that the function becomes very flat and insensitive to small changes in its input.

Another commonly encountered function is the **softplus** function (Dugas *et al.,* 2001):

`$$
\zeta(x) = log(1+ exp(x)) \\
$$`

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Softplus_Function.PNG" width=600px/>
</div>
<br>

The softplus function can be useful for producing the `$\beta$` or `$\sigma$` parameter of a normal distribution because its range is `$(0, \infty)$`. It also arises commonly when manipulating expressions involving sigmoids. The name of the softplus function comes from the fact that it is a smoothed or “softened” version of

`$$
x^{+} = max(0,x) \\
$$`

The following properties are all useful:

`$$
\sigma(x) = \frac{exp(x)}{exp(x)+exp(0)} \\
\frac{d}{dx}\sigma(x) = \sigma(x)(1-\sigma(x)) \\
1 - \sigma(x) = \sigma(-x) \\
log \sigma(x) = -\zeta(-x) \\
\frac{d}{dx}\zeta(x) = \sigma(x) \\
\forall x \in (0,1), \sigma^{-1}(x) = log \left( \frac{x}{1-x} \right) \\
\forall x > 0, \zeta^{-1}(x) = log(exp(x)-1) \\
\zeta(x) = \int_{-\infty}^{x}\sigma(y)dy \\
\zeta(x)-\zeta(-x) = x \\
$$`

## Bayes’ Rule

We often find ourselves in a situation where we know `$P(\mathrm{y}|\mathrm{x})$` and need to know `$P(\mathrm{x}|\mathrm{y})$`. Fortunately, if we also know `$P(\mathrm{x})$`, we can compute the desired quantity using **Bayes’ rule**:

`$$
P(\mathrm{x}|\mathrm{y}) = \frac{P(\mathrm{x})P(\mathrm{y}|\mathrm{x})}{P(\mathrm{y})} \\
$$`

Note that while `$P(\mathrm{y})$` appears in the formula, it is usually feasible to compute `$P(\mathrm{y})=$` `$\sum_{x}P(\mathrm{y}|\mathrm{x})P(\mathrm{x})$`, so we do not need to begin with knowledge of `$P(\mathrm{y})$`.

> 我们经常会需要在已知 `$P(\mathrm{y}|\mathrm{x})$` 时计算 `$P(\mathrm{x}|\mathrm{y})$`. 如果还知道 `$P(\mathrm{x})$`, 我们可以用**贝叶斯规则(Bayes’ rule)** 来实现这一目的:
> `$$ P(\mathrm{x}|\mathrm{y}) = \frac{P(\mathrm{x})P(\mathrm{y}|\mathrm{x})}{P(\mathrm{y})} \\ $$`
> 注意到 `$P(\mathrm{y})$` 出现在上面的公式中, 它通常使用 `$P(\mathrm{y})=\sum_{x}P(\mathrm{y}|\mathrm{x})P(\mathrm{x})$` 来计算, 所以我们并不需要事先知道 `$P(\mathrm{y})$` 的信息.

## Information Theory

Information theory is a branch of applied mathematics that revolves around quantifying how much information is present in a signal. It was originally invented to study sending messages from discrete alphabets over a noisy channel, such as communication via radio transmission. In this context, information theory tells how to design optimal codes and calculate the expected length of messages sampled from specific probability distributions using various encoding schemes. In the context of machine learning, we can also apply information theory to continuous variables where some of these message length interpretations do not apply. This field is fundamental to many areas of electrical engineering and computer science.

The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. We would like to quantify information in a way that formalizes this intuition. Specifically,

* Likely events should have low information content, and in the extreme case, events that are guaranteed to happen should have no information content whatsoever.
* Less likely events should have higher information content.
* Independent events should have additive information. For example, finding out that a tossed coin has come up as heads twice should convey twice as much information as finding out that a tossed coin has come up as heads once.

> **信息论(information theory)** 是应用数学的一个分支, 主要研究的是对一个信号包含信息的多少进行量化. 信息论的基本想法是一个不太可能的事件居然发生了, 要比一个非常可能的事件发生, 能提供更多的信息. 我们想要通过这种基本想法来量化信息:
> * 非常可能发生的事件信息量要比较少, 并且极端情况下, 确保能够发生的事件应该没有信息量.
> * 较不可能发生的事件具有更高的信息量.
> * 独立事件应具有增量的信息. 例如, 投掷的硬币两次正面朝上传递的信息量, 应该是投掷一次硬币正面朝上的信息量的两倍.

In order to satisfy all three of these properties, we define the **self-information** of an event `$\mathrm{x}=x$` to be

`$$
I(x) = -log P(x) \\
$$`

we always use log to mean the natural logarithm, with base `$e$`. Our definition of `$I(x)$` is therefore written in units of **nats**. One nat is the amount of information gained by observing an event of probability `$\frac{1}{e}$` . Other texts use base-2 logarithms and units called **bits** or **shannons**; information measured in bits is just a rescaling of information measured in nats.

> 为了满足上述三个性质, 我们定义一个事件 `$\mathrm{x}=x$` 的**自信息(self-information)** 为:
> `$$ I(x) = -log P(x) \\ $$`
> 我们定义的 `$I(x)$` 单位是 **奈特(nats)** . 一奈特是以 `$\frac{1}{e}$` 的概率观测到一个事件时获得的信息量.

When `$\mathrm{x}$` is continuous, we use the same definition of information by analogy, but some of the properties from the discrete case are lost. For example, an event with unit density still has zero information, despite not being an event that is guaranteed to occur.

Self-information deals only with a single outcome. We can quantify the amount of uncertainty in an entire probability distribution using the **Shannon entropy**:

`$$
H(x) = E_{\mathrm{x} \sim P}[I(x)] = -E_{\mathrm{x} \sim P}[log P(x)] \\
$$`

also denoted `$H(P)$`. In other words, the Shannon entropy of a distribution is the expected amount of information in an event drawn from that distribution. It gives a lower bound on the number of bits (if the logarithm is base 2, otherwise the units are different) needed on average to encode symbols drawn from a distribution `$P$`. Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy. When `$x$` is continuous, the Shannon entropy is known as the **differential entropy**.


> 我们可以用 **香农熵(Shannon entropy)** 来对整个概率分布中的不确定性总量进行量化:
> `$$ H(x) = E_{\mathrm{x} \sim P}[I(x)] = -E_{\mathrm{x} \sim P}[log P(x)] \\ $$`
> 也记作 `$H(P)$`. 换言之, 一个分布的香农熵是指遵循这个分布的事件所产生的期望信息总量. 那些接近确定性的分布 (输出几乎可以确定) 具有较低的熵; 那些接近均匀分布的概率分布具有较高的熵. 当 `$x$` 是连续的, 香农熵被称为**微分熵(differential entropy)** .

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_02_Shannon_Entropy.PNG" width=600px/>
</div>
<br>

This plot shows how distributions that are closer to deterministic have low Shannon entropy while distributions that are close to uniform have high Shannon entropy. On the horizontal axis, we plot `$p$`, the probability of a binary random variable being equal to 1. The entropy is given by `$(p−1)log(1−p)−plog(p)$`. When `$p$` is near 0,the distribution is nearly deterministic, because the random variable is nearly always 0. When `$p$` is near 1, the distribution is nearly deterministic, because the random variable is nearly always 1. When `$p=0.5$`, the entropy is maximal, because the distribution is uniform over the two outcomes.

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/prob.html.
