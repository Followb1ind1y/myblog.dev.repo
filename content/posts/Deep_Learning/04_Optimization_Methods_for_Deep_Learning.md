---
title: "Optimization Methods for Deep Learning"
date: "2021-07-27"
tags: ["Gradient Descent", "Stochastic Gradient Descent", "Mini-batch Gradient Descent", "Momentum", "Nesterov Momentum", "AdaGrad", "RMSProp", "Adam"]
categories: ["Deep Learning", "Data Science", "Optimization"]
weight: 3
---

## Gradient Descent Optimization

Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function `$f(x)$` by altering `$x$`. We usually phrase most optimization problems in terms of minimizing `$f(x)$`. Maximization may be accomplished via a minimization algorithm by minimizing `$-f(x)$`.

The function we want to minimize or maximize is called the **objective function** or **criterion**. When we are minimizing it, we may also call it the **cost function**, **loss function**, or **error function**.

>大多数深度学习算法都涉及某种形式的优化. 优化指的是改变 `$x$` 以最小化或最大化某个函数 `$f(x)$` 的任务. 我们通常以最小化 `$f(x)$` 指代大多数最优化问题. 我们把要最小化或最大化的函数称为 **目标函数(objective function)** 或**准则 (criterion)**.当我们对其进行最小化时,我们也把它称为 **代价函数(cost function)**、**损失函数(loss function)** 或 **误差函数(error function)**.

We often denote the value that minimizes or maximizes a function with a superscript `$\ast$` . For example, we might say `$x^{*}=\arg \min f(x)$`.

Suppose we have a function `$y = f(x)$`, where both `$x$` and `$y$` are real numbers. The **derivative** of this function is denoted as `$f'(x)$` or as `$\frac{dy}{dx}$` . The derivative `$f'(x)$` gives the slope of `$f(x)$` at the point `$x$`. In other words, it specifies how to scale a small change in the input in order to obtain the corresponding change in the output: `$f(x+\epsilon) \approx f(x) \ +$` `$ \epsilon f'(x)$`.

The derivative is therefore useful for minimizing a function because it tells us how to change `$x$` in order to make a small improvement in `$y$`. For example, we know that `$f(x-\epsilon \mathrm{sign}(f'(x)))$` is less than `$f(x)$` for small enough `$\epsilon$`. We can thus reduce `$f(x)$` by moving `$x$` in small steps with opposite sign of the derivative. This technique is called **gradient descent**.

<div align="center">
  <img src="/img_DL/ML_Basics_03_Gradient_Descent.PNG" width=500px/>
</div>
<br>

When `$f'(x)=0$`, the derivative provides no information about which direction to move. Points where `$f'(x)=0$` are known as **critical points** or **stationary points**. A **local minimum** is a point where `$f(x)$` is lower than at all neighboring points, so it is no longer possible to decrease `$f(x)$` by making infinitesimal steps. A **local maximum** is a point where `$f(x)$` is higher than at all neighboring points, so it is not possible to increase `$f(x)$` by making infinitesimal steps. Some critical points are neither maxima nor minima. These are known as **saddle points**.

> 当 `$f'(x)=0$`,导数无法提供往哪个方向移动的信息.`$f'(x)=0$` 的点称为 **临界点(critical point)** 或 **驻点(stationary point)**.一个 **局部极小点(local minimum)** 意味着这个点的 `$f(x)$` 小于所有邻近点,因此不可能通过移动无穷小的步长来减小 `$f(x)$`.一个 **局部极大点(local maximum)** 意味着这个点的 `$f(x)$` 大于所有邻近点,因此不可能通过移动无穷小的步长来增大 `$f(x)$`.有些临界点既不是最小点也不是最大点.这些点被称为 **鞍点(saddle point)**.

<div align="center">
  <img src="/img_DL/ML_Basics_03_Critical_Points.PNG" width=500px/>
</div>
<br>

A point that obtains the absolute lowest value of `$f(x)$` is a **global minimum**. It is possible for there to be only one global minimum or multiple global minima of the function. It is also possible for there to be local minima that are not globally optimal. In the context of deep learning, we optimize functions that may have many local minima that are not optimal, and many saddle points surrounded by very flat regions. All of this makes optimization very difficult, especially when the input to the function is multidimensional. We therefore usually settle for finding a value of f that is very low, but not necessarily minimal in any formal sense.

>使 `$f(x)$` 取得绝对的最小值(相对所有其他值)的点是 **全局最小点(global minimum)**.函数可能只有一个全局最小点或存在多个全局最小点,还可能存在不是全局最优的局部极小点.

<div align="center">
  <img src="/img_DL/ML_Basics_03_Global_Minimum.PNG" width=500px/>
</div>
<br>


We often minimize functions that have multiple inputs: `$f: \mathbb{R}^{n} \to \mathbb{R}$`. For the concept of “minimization” to make sense, there must still be only one (scalar) output.

For functions with multiple inputs, we must make use of the concept of **partial derivatives**. The partial derivative `$\frac{\partial}{\partial x_{i}}$` measures how `$f$` changes as only the variable `$x_{i}$` increases at point `$x$`. The **gradient** generalizes the notion of derivative to the case where the derivative is with respect to a vector: the gradient of `$f$` is the vector containing all of the partial derivatives, denoted `$\nabla_{x} \ f(x)$`. Element `$i$` of the gradient is the partial derivative of `$f$` with respect to `$x_{i}$`. In multiple dimensions, critical points are points where every element of the gradient is equal to zero.

>  **梯度(gradient)** 是相对一个向量求导的导数: `$f$` 的导数是包含所有偏导数的向量，记为 `$\nabla_{x} \ f(x)$`。在多维情况下，临界点是梯度中所有元素都为零的点。

The **directional derivative** in direction `$u$` (a unit vector) is the slope of the function `$f$` in direction `$u$`. In other words, the directional derivative is the derivative of the function `$f(x+\alpha u)$` with respect to `$\alpha$`, evaluated at `$\alpha=0$`. Using the chain rule, we can see that `$\frac{\partial}{\partial \alpha}f(x+\alpha u)$` evaluates to `$u^{T}\nabla_{x} \ f(x)$` when `$\alpha=0$`.

To minimize `$f$`, we would like to find the direction in which `$f$` decreases the fastest. We can do this using the directional derivative:

`$$
\min_{u,u^{T}u=1} u^{T}\nabla_{x} \ f(x) \\
= \min_{u,u^{T}u=1} \lVert u \rVert_{2} \lVert \nabla_{x} \ f(x) \rVert_{2} \cos \theta \\
$$`


where `$\theta$` is the angle between `$u$` and the gradient. Substituting in `$\lVert u \rVert_{2}=1$` and ignoring factors that do not depend on `$u$`, this simplifies to `$\min_{u} \cos \theta$`. This is minimized when `$u$` points in the opposite direction as the gradient. In other words, the gradient points directly uphill, and the negative gradient points directly downhill. We can decrease `$f$` by moving in the direction of the negative gradient. This is known as the **method of steepest descent or gradient descent**.

Steepest descent proposes a new point

`$$
x' = x - \epsilon \nabla_{x} \ f(x) \\
$$`

where `$\epsilon$` is the learning rate, a positive scalar determining the size of the step. We can choose `$\epsilon$` in several different ways. A popular approach is to set `$\epsilon$` to a small constant. Sometimes, we can solve for the step size that makes the directional derivative vanish. Another approach is to evaluate `$f(x - \epsilon \nabla_{x} \ f(x))$` for several values of `$\epsilon$` and choose the one that results in the smallest objective function value. This last strategy is called a **line search**.

Steepest descent converges when every element of the gradient is zero (or, in practice, very close to zero). In some cases, we may be able to avoid running this iterative algorithm, and just jump directly to the critical point by solving the equation `$\nabla_{x} \ f(x)=0$` for `$x$`.

Although gradient descent is limited to optimization in continuous spaces, the general concept of repeatedly making a small move (that is approximately the best small move) towards better configurations can be generalized to discrete spaces. Ascending an objective function of discrete parameters is called **hill climbing**.


## Gradient Descent Variants

There are three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.

### Batch gradient descent

Vanilla gradient descent, aka **batch gradient descent**, computes the gradient of the cost function w.r.t. to the parameters `$\theta$` for the entire training dataset:

`$$
\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta) \\
$$`

As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient descent can be very slow and is intractable for datasets that don't fit in memory. Batch gradient descent also doesn't allow us to update our model online, i.e. with new examples on-the-fly.

> **BGD** 为梯度下降算法中最基础的一个算法，其损失函数定义如下：
`$$
J(\theta) = \frac{1}{2m}\sum_{i=1}^{m} \left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\
$$`
针对任意参数`$\theta_{j}$`我们可以求得其梯度为：
`$$
\nabla_{\theta}J(\theta) = \frac{\partial J(\theta)}{\partial \theta_{j}} = -\frac{1}{m}\sum_{i=1}^{m}\left(y^{(i)}-h_{\theta}\left(x^{(i)}\right)\right)x_{j}^{(i)}  \\
$$`
之后，对于任意参数`$\theta_{j}$`我们按照其负梯度方向进行更新：
`$$
\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta) \\
$$`
从上述算法流程中我们可以看到，BGD 算法每次计算梯度都使用了**整个训练集**，也就是说对于给定的一个初始点，其每一步的更新都是沿着全局梯度最大的负方向。但这同样是其问题，当`$m$`太大时，整个算法的计算开销就很高了

### Stochastic gradient descent

**Stochastic gradient descent (SGD)** in contrast performs a parameter update for each training example `$x^{(i)}$` and label `$y^{(i)}$`:

`$$
\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta;x^{(i)};y^{(i)}) \\
$$`

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be used to learn online. SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily.

While batch gradient descent converges to the minimum of the basin the parameters are placed in, SGD's fluctuation, on the one hand, enables it to jump to new and potentially better local minima. On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting. However, it has been shown that when we slowly decrease the learning rate, **SGD shows the same convergence behaviour as batch gradient descent**, almost certainly converging to a local or the global minimum for non-convex and convex optimization respectively.

> **SGD** 相比于 BGD，其最主要的区别就在于计算梯度时不再利用整个数据集，而是针对 **单个样本** 计算梯度并更新权重，因此，其损失函数定义如下：
`$$
J(\theta;x^{(i)};y^{(i)}) = \frac{1}{2}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) \\
$$`
之后，我们按照其负梯度方向进行更新：
`$$
\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta;x^{(i)};y^{(i)}) \\
$$`
SGD 相比于 BGD 具有训练速度快的优势，但同时由于权重改变的方向并不是全局梯度最大的负方向，甚至相反，因此不能够保证每次损失函数都会减小。

### Mini-batch gradient descent

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of `$b$` training examples:

`$$
\theta = \theta - \eta \cdot \nabla_{\theta}J(\theta;x^{(i:i+b)};y^{(i:i+b)}) \\
$$`

This way, it reduces the variance of the parameter updates, which can lead to more stable convergence; and can make use of highly optimized matrix optimizations common to state-of-the-art deep learning libraries that make computing the gradient w.r.t. a mini-batch very efficient. Common mini-batch sizes range between 50 and 256, but can vary for different applications. Mini-batch gradient descent is typically the algorithm of choice when training a neural network and the term SGD usually is employed also when mini-batches are used.

> 针对 BGD 和 SGD 的问题，**MBGD** 则是一个折中的方案，在每次更新参数时，MBGD会选取 `$b$`个样本计算梯度并更新权重。常见的小批量大小范围在 50 到 256 之间，但可能因不同的应用程序而异。MBGD通常是训练神经网络时选择的算法。

## Challenges in Neural Network Optimization

Optimization in general is an extremely difficult task. Traditionally, machine learning has avoided the difficulty of general optimization by carefully designing the objective function and constraints to ensure that the optimization problem is convex. When training neural networks, we must confront the general non-convex case. Even convex optimization is not without its complications.

Mini-batch gradient descent, however, does not guarantee good convergence, but offers a few challenges that need to be addressed:

* Choosing a proper learning rate can be difficult. A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.

* Additionally, the same learning rate applies to all parameter updates. If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.

* Another key challenge of minimizing highly non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima. Dauphin et al. argue that the difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

> 梯度下降可能会遇到的问题和挑战：
> * 选择合适的学习率可能很困难。学习率太小会导致收敛速度很慢，而学习率太大会阻碍收敛并导致损失函数在最小值附近波动甚至发散。
> * 相同的学习率适用于所有参数更新。如果我们的数据是稀疏的并且我们的特征具有非常不同的频率，我们可能不想将它们全部更新到相同的程度。
> * 最小化神经网络常见的另一个关键挑战是避免陷入其众多次优局部最小值和鞍点。

## Gradient Descent Optimization Algorithms

### Momentum

While stochastic gradient descent remains a very popular optimization strategy, learning with it can sometimes be slow. The method of **Momentum** (Polyak, 1964) is designed to accelerate learning, especially in the face of high curvature, small but consistent gradients, or noisy gradients. The momentum algorithm accumulates an exponentially decaying moving average of past gradients and continues to move in their direction.

<div align="center">
  <img src="/img_DL/03_Momentum.PNG" width=400px/>
</div>
<br>

Formally, the momentum algorithm introduces a variable `$v$` that plays the role of velocity - it is the direction and speed at which the parameters move through parameter space. The velocity is set to an exponentially decaying average of the negative gradient. Momentum helps accelerate SGD in the relevant direction. It does this by adding a fraction `$\gamma$` of the update vector of the past time step to the current update vector:

`$$
\begin{align*}
v_{t} &= - \eta\nabla_{\theta}J(\theta_{t}) + \gamma v_{t-1} \\
\theta_{t} &= \theta_{t-1} + v_{t}
\end{align*}
$$`

Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way. The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.

> 当梯度沿着一个方向要明显比其他方向陡峭，我们可以形象的称之为峡谷形梯度，这种情况多位于局部最优点附近。在这种情况下，SGD 通常会摇摆着通过峡谷的斜坡，这就导致了其到达局部最优值的速度过慢。因此，针对这种情况，**Momentum(动量)** 方法提供了一种解决方案。针对原始的 SGD 算法，参数每 `$t$` 步的变化量可以表示为
`$$
v_{t} = - \eta\nabla_{\theta}J(\theta_{t}) \\
$$`
Momentum 算法则在其变化量中添加了一个动量分量，即
`$$
\begin{align*}
v_{t} &= - \eta\nabla_{\theta}J(\theta_{t}) + \gamma v_{t-1} \\
\theta_{t} &= \theta_{t-1} + v_{t}
\end{align*}
$$`
对于添加的动量项，当第 `$t$` 步和第 `$t-1$` 步的梯度方向 **相同** 时， `$\theta$`则以更快的速度更新；当第 `$t$` 步和第 `$t-1$` 步的梯度方向 **相反** 时， `$\theta$`则以较慢的速度更新。

### Nesterov Momentum

Sutskever et al. (2013) introduced a variant of the momentum algorithm that was inspired by **Nesterov’s accelerated gradient method** (Nesterov, 1983, 2004). The update rules in this case are given by:

`$$
\begin{align*}
v_{t} &= - \eta\nabla_{\theta}J(\theta_{t}+\gamma v_{t-1}) + \gamma v_{t-1} \\
\theta_{t} &= \theta_{t-1} + v_{t}
\end{align*}
$$`


where the parameters `$\gamma$` and `$\eta$` play a similar role as in the standard momentum method. The difference between Nesterov momentum and standard momentum is where the gradient is evaluated. With Nesterov momentum the gradient is evaluated after the current velocity is applied. Thus one can interpret Nesterov momentum as attempting to add a **correction factor** to the standard method of momentum.

We  usually set the momentum term `$\gamma$` to a value of around 0.9. While Momentum first computes the current gradient (small blue vector) and then takes a big jump in the direction of the updated accumulated gradient (big blue vector), NAG first makes a big jump in the direction of the previous accumulated gradient (brown vector), measures the gradient and then makes a correction (red vector), which results in the complete NAG update (green vector). This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks.

<div align="center">
  <img src="/img_DL/04_Nesterov_Momentum.PNG" width=400px/>
</div>
<br>

> **NAG (Nesterov Accelerated Gradient)** 是一种 Momentum 算法的变种，其核心思想会利用“下一步的梯度”确定“这一步的梯度”，当然这里“下一步的梯度”并非真正的下一步的梯度，而是指仅根据动量项更新后位置的梯度。
`$$
\begin{align*}
v_{t} &= - \eta\nabla_{\theta}J(\theta_{t}+\gamma v_{t-1}) + \gamma v_{t-1} \\
\theta_{t} &= \theta_{t-1} + v_{t}
\end{align*}
$$`
针对 Momentum 和 NAG 两种不同的方法，其更新权重的差异如下图所示：

<div align="center">
  <img src="/img_DL/04_Nesterov_Momentum_Diff.PNG" width=600px/>
</div>
<br>

### AdaGrad

**AdaGrad** is an algorithm for gradient-based optimization that does just this: It adapts the learning rate to the parameters, performing smaller updates
(i.e. low learning rates) for parameters associated with frequently occurring features, and larger updates (i.e. high learning rates) for parameters associated with infrequent features. For this reason, it is well-suited for dealing with sparse data.

Previously, we performed an update for all parameters `$\theta$` at once as every parameter `$\theta_{i}$` used the same learning rate `$\eta$`. As AdaGrad uses a different learning rate for every parameter `$\theta_{i}$` at every time step `$t$`, we first show AdaGrad's per-parameter update, which we then vectorize. For brevity, we use `$g_{t}$` to denote the gradient at time step `$t$`. `$g_{t,i}$` is then the partial derivative of the objective function w.r.t. to the parameter `$\theta_{i}$` at time step `$t$`:

`$$
g_{t,i} = \nabla_{\theta}J(\theta_{t,i}) \\
$$`

The SGD update for every parameter `$\theta_{i}$` at each time step `$t$` then becomes:

`$$
\theta_{t+1,i} = \theta_{t,i} - \eta \cdot g_{t,i} \\
$$`

In its update rule, AdaGrad modifies the general learning rate `$\eta$`  at each time step `$t$` for every parameter `$\theta_{i}$` based on the past gradients that have been computed for `$\theta_{i}$`:

`$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii}+\epsilon}} \cdot g_{t,i} \\
$$`

`$G_{t} \in \mathbb{R}^{d \times d}$` here is a diagonal matrix where each diagonal element `$i$`, `$i$` is the sum of the squares of the gradients w.r.t. `$\theta_{i}$` up to time step `$t$`, while `$\epsilon$` is a smoothing term that avoids division by zero (usually on the order of 1e-8). Interestingly, without the square root operation, the algorithm performs much worse.

`$$
G_{t+1} = G_{t} + g \odot g \\
$$`

As `$G_{t}$` contains the sum of the squares of the past gradients w.r.t. to all parameters `$\theta$` along its diagonal, we can now vectorize our implementation by performing a matrix-vector product `$\odot$` between `$G_{t}$` and `$g_{t}$`:

`$$
\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t} \\
$$`

One of AdaGrad's main benefits is that it **eliminates the need to manually tune the learning rate**. Most implementations use a default value of 0.01 and leave it at that.

AdaGrad's main weakness is its accumulation of the squared gradients in the denominator: Since every added term is positive, the accumulated sum keeps growing during training. This in turn causes the learning rate to shrink and eventually become infinitesimally small, at which point the algorithm is no longer able to acquire additional knowledge.

> **AdaGrad** 是一种具有自适应学习率的的方法，其对于低频特征的参数选择更大的更新量，对于高频特征的参数选择更小的更新量。因此，AdaGrad算法更加适用于处理稀疏数据。

### RMSProp

The **RMSProp** algorithm (Hinton, 2012) modifies AdaGrad to perform better in the non-convex setting by changing the gradient accumulation into an exponentially weighted moving average. AdaGrad is designed to converge rapidly when applied to a convex function. When applied to a non-convex function to train a neural network, the learning trajectory may pass through many different structures and eventually arrive at a region that is a locally convex bowl. AdaGrad shrinks the learning rate according to the entire history of the squared gradient and may have made the learning rate too small before arriving at such a convex structure. RMSProp uses an exponentially decaying average to discard history from the extreme past so that it can converge rapidly after finding a convex bowl, as if it were an instance of the AdaGrad algorithm initialized within that bowl.

Compared to AdaGrad, the use of the moving average introduces a new hyperparameter, `$\rho$`, that controls the length scale of the moving average. Hinton suggests `$\rho$` to be set to 0.9, while a good default value for the learning rate `$\eta$` is 0.001.

`$$
\begin{align*}
G_{t} &= \rho G_{t-1} + (1-\rho) g \odot g \\
\theta_{t+1} &= \theta_{t} - \frac{\eta}{\sqrt{G_{t}+\epsilon}} \odot g_{t} \\
\end{align*}
$$`

Empirically, RMSProp has been shown to be an effective and practical optimization algorithm for deep neural networks. It is currently one of the go-to optimization methods being employed routinely by deep learning practitioners.

### Adam

**Adaptive Moment Estimation (Adam)** is another method that computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients `$v_{t}$` like RMSprop, Adam also keeps an exponentially decaying average of past gradients `$m_{t}$`, similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface. We compute the decaying averages of past and past squared gradients `$m_{t}$` and `$v_{t}$` respectively as follows:

`$$
\begin{align*}
m_{t} &= \beta_{1}m_{t-1} + (1-\beta_{1})g_{t} \\
v_{t} &= \beta_{2}v_{t-1} + (1-\beta_{2})g_{t}^{2} \\
\end{align*}
$$`

`$m_{t}$` and `$v_{t}$` are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively, hence the name of the method. As `$m_{t}$` and `$v_{t}$` are initialized as vectors of 0's, the authors of Adam observe that they are biased towards zero, especially during the initial time steps, and especially when the decay rates are small (i.e. `$\beta_{1}$` and `$\beta_{2}$` are close to 1).

`$$
\theta_{t+1} = \theta_{t} - \frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon}\hat{m}_{t} \\
$$`

The authors propose default values of 0.9 for `$\beta_{1}$`, 0.999 for `$\beta_{2}$`, and `$10^{-8}$`for `$\epsilon$`. They show empirically that Adam works well in practice and compares favorably to other adaptive learning-method algorithms.

In summary, RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numinator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. show that its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, **Adam might be the best overall choice**.

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/optimization.html.

[2] Ruder, S. (2020, March 20). An overview of gradient descent optimization algorithms. https://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants.
