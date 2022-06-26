---
title: "[ML Basics] Numerical Computation"
date: "2021-06-28"
tags: ["Numerical Computation"]
categories: ["Machine Learning Basics", "Data Science"]
weight: 3
---

## Overflow and Underflow

The fundamental difficulty in performing continuous math on a digital computer is that we need to represent infinitely many real numbers with a finite number of bit patterns. This means that for almost all real numbers, we incur some approximation error when we represent the number in the computer. In many cases, this is just rounding error. Rounding error is problematic, especially when it compounds across many operations, and can cause algorithms that work in theory to fail in practice if they are not designed to minimize the accumulation of rounding error.

One form of rounding error that is particularly devastating is **underflow**. Underflow occurs when numbers near zero are rounded to zero. Many functions behave qualitatively differently when their argument is zero rather than a small positive number. For example, we usually want to avoid division by zero (some software environments will raise exceptions when this occurs, others will return a result with a placeholder not-a-number value) or taking the logarithm of zero (this is usually treated as `$-\infty$`, which then becomes not-a-number if it is used for many further arithmetic operations).

Another highly damaging form of numerical error is **overflow**. Overflow occurs when numbers with large magnitude are approximated as `$\infty$` or `$-\infty$`. Further arithmetic will usually change these infinite values into not-a-number values.

> 一种极具毁灭性的舍入误差是 **下溢(underflow)**. 当接近零的数被四舍五入为零时发生下溢. 许多函数在其参数为零而不是一个很小的正数时才会表现出质的不同.例如, 我们通常要避免被零除(一些软件环境将在这种情况下抛出异常, 有些会返回一个非数字 (not-a-number, NaN) 的占位符)或避免取零的对数(这通常被 视为 `$-\infty$`, 进一步的算术运算会使其变成非数字).
>
>另一个极具破坏力的数值错误形式是 **上溢(overflow)**. 当大量级的数被近似为 `$\infty$` 或 `$-\infty$` 时发生上溢. 进一步的运算通常会导致这些无限值变为非数字.

## Poor Conditioning

Conditioning refers to how rapidly a function changes with respect to small changes in its inputs. Functions that change rapidly when their inputs are perturbed slightly can be problematic for scientific computation because rounding errors in the inputs can result in large changes in the output.

Consider the function `$f(x) = A^{−1}x$`. When `$A \in \mathbb{R}^{n \times n}$` has an eigenvalue decomposition, its **condition number** is

`$$
\max_{i,j}|\frac{\lambda_{i}}{\lambda_{j}}|
$$`

This is the ratio of the magnitude of the largest and smallest eigenvalue. When this number is large, matrix inversion is particularly sensitive to error in the input.

This sensitivity is an intrinsic property of the matrix itself, not the result of rounding error during matrix inversion. Poorly conditioned matrices amplify pre-existing errors when we multiply by the true matrix inverse. In practice, the error will be compounded further by numerical errors in the inversion process itself.

## Gradient-Based Optimization

Most deep learning algorithms involve optimization of some sort. Optimization refers to the task of either minimizing or maximizing some function `$f(x)$` by altering `$x$`. We usually phrase most optimization problems in terms of minimizing `$f(x)$`. Maximization may be accomplished via a minimization algorithm by minimizing `$-f(x)$`.

The function we want to minimize or maximize is called the **objective function** or **criterion**. When we are minimizing it, we may also call it the **cost function**, **loss function**, or **error function**.

>大多数深度学习算法都涉及某种形式的优化. 优化指的是改变 `$x$` 以最小化或最大化某个函数 `$f(x)$` 的任务. 我们通常以最小化 `$f(x)$` 指代大多数最优化问题. 我们把要最小化或最大化的函数称为 **目标函数(objective function)** 或**准则 (criterion)**.当我们对其进行最小化时,我们也把它称为 **代价函数(cost function)**、**损失函数(loss function)** 或 **误差函数(error function)**.

We often denote the value that minimizes or maximizes a function with a superscript `$\ast$` . For example, we might say `$x^{*}=\arg \min f(x)$`.

Suppose we have a function `$y = f(x)$`, where both `$x$` and `$y$` are real numbers. The **derivative** of this function is denoted as `$f'(x)$` or as `$\frac{dy}{dx}$` . The derivative `$f'(x)$` gives the slope of `$f(x)$` at the point `$x$`. In other words, it specifies how to scale a small change in the input in order to obtain the corresponding change in the output: `$f(x+\epsilon) \approx f(x) \ +$` `$ \epsilon f'(x)$`.

The derivative is therefore useful for minimizing a function because it tells us how to change `$x$` in order to make a small improvement in `$y$`. For example, we know that `$f(x-\epsilon \mathrm{sign}(f'(x)))$` is less than `$f(x)$` for small enough `$\epsilon$`. We can thus reduce `$f(x)$` by moving `$x$` in small steps with opposite sign of the derivative. This technique is called **gradient descent**.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_03_Gradient_Descent.PNG" width=500px/>
</div>
<br>

When `$f'(x)=0$`, the derivative provides no information about which direction to move. Points where `$f'(x)=0$` are known as **critical points** or **stationary points**. A **local minimum** is a point where `$f(x)$` is lower than at all neighboring points, so it is no longer possible to decrease `$f(x)$` by making infinitesimal steps. A **local maximum** is a point where `$f(x)$` is higher than at all neighboring points, so it is not possible to increase `$f(x)$` by making infinitesimal steps. Some critical points are neither maxima nor minima. These are known as **saddle points**.

> 当 `$f'(x)=0$`,导数无法提供往哪个方向移动的信息.`$f'(x)=0$` 的点称为 **临界点(critical point)** 或 **驻点(stationary point)**.一个 **局部极小点(local minimum)** 意味着这个点的 `$f(x)$` 小于所有邻近点,因此不可能通过移动无穷小的步长来减小 `$f(x)$`.一个 **局部极大点(local maximum)** 意味着这个点的 `$f(x)$` 大于所有邻近点,因此不可能通过移动无穷小的步长来增大 `$f(x)$`.有些临界点既不是最小点也不是最大点.这些点被称为 **鞍点(saddle point)**.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_03_Critical_Points.PNG" width=500px/>
</div>
<br>

A point that obtains the absolute lowest value of `$f(x)$` is a **global minimum**. It is possible for there to be only one global minimum or multiple global minima of the function. It is also possible for there to be local minima that are not globally optimal. In the context of deep learning, we optimize functions that may have many local minima that are not optimal, and many saddle points surrounded by very flat regions. All of this makes optimization very difficult, especially when the input to the function is multidimensional. We therefore usually settle for finding a value of f that is very low, but not necessarily minimal in any formal sense.

>使 `$f(x)$` 取得绝对的最小值(相对所有其他值)的点是 **全局最小点(global minimum)**.函数可能只有一个全局最小点或存在多个全局最小点,还可能存在不是全局最优的局部极小点.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_03_Global_Minimum.PNG" width=500px/>
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

## Jacobian Hessian Matrices / Newton's Methods

Sometimes we need to find all of the partial derivatives of a function whose input and output are both vectors. The matrix containing all such partial derivatives is known as a **Jacobian matrix**. Specifically, if we have a function `$f: \mathbb{R}^{m} \to \mathbb{R}^{n}$`, then the Jacobian matrix `$J \in  \mathbb{R}^{n \times m}$` of `$f$` is defined such that `$J_{i,j}=\frac{\partial}{\partial x_{j}}f(x)_{i}$`.

`$$
J = \begin{bmatrix}
\frac{\partial f}{\partial x_{1}} & \cdots & \frac{\partial f}{\partial x_{n}} \\
\end{bmatrix} =  
\begin{bmatrix}
\nabla^{T}f_{1} \\
\vdots \\
\nabla^{T}f_{m} \\
\end{bmatrix} =
\begin{bmatrix}
\frac{\partial f_{1}}{\partial x_{1}} & \cdots & \frac{\partial f_{1}}{\partial x_{n}} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_{m}}{\partial x_{1}} & \cdots & \frac{\partial f_{m}}{\partial x_{n}} \\
\end{bmatrix} \\
$$`

> 有时我们需要计算输入和输出都为向量的函数的所有偏导数.包含所有这样的偏导数的矩阵被称为 **Jacobian 矩阵**.

We are also sometimes interested in a derivative of a derivative. This is known as a second derivative. For example, for a function `$f: \mathbb{R}^{n} \to \mathbb{R}$`, the derivative with respect to `$x_{i}$` of the derivative of `$f$` with respect to `$x_{j}$` is denoted as `$\frac{\partial^{2}}{\partial x_{i}\partial x_{j}}f$`. In a single dimension, we can denote `$\frac{d^{2}}{dx^{2}}f$` by `$f''(x)$`. The second derivative tells us how the first derivative will change as we vary the input. This is important because it tells us whether a gradient step will cause as much of an improvement as we would expect based on the gradient alone. We can think of the second derivative as measuring **curvature**.

When our function has multiple input dimensions, there are many second derivatives. These derivatives can be collected together into a matrix called the **Hessian matrix**. The Hessian matrix `$H(f)(x)$` is defined such that `$H(f)(x)_{i,j} = \frac{\partial^{2}}{\partial x_{i}\partial x_{j}}f(x)$`

`$$
H(f)(x)_{i,j} =
\begin{bmatrix}
\frac{\partial^{2} f}{\partial x_{1}^{2}} & \frac{\partial^{2} f}{\partial x_{1}\partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{1}\partial x_{n}} \\
\frac{\partial^{2} f}{\partial x_{2}\partial x_{1}} & \frac{\partial^{2} f}{\partial x_{2}^{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{2}\partial x_{n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^{2} f}{\partial x_{n}\partial x_{1}} & \frac{\partial^{2} f}{\partial x_{n}\partial x_{2}} & \cdots & \frac{\partial^{2} f}{\partial x_{n}^{2}} \\
\end{bmatrix} \\
$$`

Equivalently, the Hessian is the Jacobian of the gradient.

> 当我们的函数具有多维输入时,二阶导数也有很多.我们可以将这些导数合并成一个矩阵,称为 **Hessian 矩阵**.

Anywhere that the second partial derivatives are continuous, the differential operators are commutative, i.e. their order can be swapped:

`$$
\frac{\partial^{2}}{\partial x_{i}\partial x_{j}}f(x)=\frac{\partial^{2}}{\partial x_{j}\partial x_{i}}f(x) \\
$$`

This implies that `$H_{i,j}=h_{j,i}$`, so the Hessian matrix is symmetric at such points. Most of the functions we encounter in the context of deep learning have a symmetric Hessian almost everywhere. Because the Hessian matrix is real and symmetric, we can decompose it into a set of real eigenvalues and an orthogonal basis of eigenvectors. The second derivative in a specific direction represented by a unit vector `$f$` is given by `$d^{T}Hd$`. When `$d$` is an eigenvector of `$H$` , the second derivative in that direction is given by the corresponding eigenvalue. For other directions of `$d$`, the directional second derivative is a weighted average of all of the eigenvalues, with weights between 0 and 1, and eigenvectors that have smaller angle with `$d$` receiving more weight. The maximum eigenvalue determines the maximum second derivative and the minimum eigenvalue determines the minimum second derivative.

The (directional) second derivative tells us how well we can expect a gradient descent step to perform. We can make a second-order Taylor series approximation to the function `$f(x)$` around the current point `$x^{(0)}$`:

`$$
f(x) \approx f(x^{(0)}) +(x - x^{(0)})^{\mathrm{T}}g + \frac{1}{2}(x - x^{(0)})^{\mathrm{T}}H(x - x^{(0)}) \\
$$`

where `$g$` is the gradient and H is the Hessian at `$x^{(0)}$`. If we use a learning rate of `$\epsilon$`, then the new point `$x$` will be given by `$x^{(0)}-\epsilon g$`. Substituting this into our approximation, we obtain

`$$
f(x^{(0)}-\epsilon g) \approx f(x^{(0)}) -\epsilon g^{\mathrm{T}}g + \frac{1}{2}\epsilon^{2}g^{\mathrm{T}}Hg \\
$$`

There are three terms here: the original value of the function, the expected improvement due to the slope of the function, and the correction we must apply to account for the curvature of the function. When this last term is too large, the gradient descent step can actually move uphill. When `$g^{\mathrm{T}}Hg$` is zero or negative, the Taylor series approximation predicts that increasing `$\epsilon$` forever will decrease `$f$` forever. In practice, the Taylor series is unlikely to remain accurate for large `$\epsilon$`, so one must resort to more heuristic choices of `$\epsilon$` in this case. When `$g^{\mathrm{T}}Hg$` is positive, solving for the optimal step size that decreases the Taylor series approximation of the function the most yields

`$$
\varepsilon^{*} = \frac{g^{\mathrm{T}}g}{g^{\mathrm{T}}Hg} \\
$$`

In the worst case, when `$g$` aligns with the eigenvector of `$H$` corresponding to the maximal eigenvalue `$\lambda_{\max}$`, then this optimal step size is given by `$\frac{1}{\lambda_{\max}}$` . To the extent that the function we minimize can be approximated well by a quadratic function, the eigenvalues of the Hessian thus determine the scale of the learning rate.

In multiple dimensions, there is a different second derivative for each direction at a single point. The condition number of the Hessian at this point measures how much the second derivatives differ from each other. When the Hessian has a poor condition number, gradient descent performs poorly. This is because in one direction, the derivative increases rapidly, while in another direction, it increases slowly. Gradient descent is unaware of this change in the derivative so it does not know that it needs to explore preferentially in the direction where the derivative remains negative for longer. It also makes it difficult to choose a good step size. The step size must be small enough to avoid overshooting the minimum and going uphill in directions with strong positive curvature. This usually means that the step size is too small to make significant progress in other directions with less curvature.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_03_Hessian_and_GD.PNG" width=400px/>
</div>
<br>

> 多维情况下，单个点处每个方向上的二阶导数是不同。Hessian 的条件数衡量 这些二阶导数的变化范围。当 Hessian 的条件数很差时，梯度下降法也会表现得很差。这是因为一个方向上的导数增加得很快，而在另一个方向上增加得很慢。梯度下降不知道导数的这种变化，所以它不知道应该优先探索导数长期为负的方向。病态条件也导致很难选择合适的步长。

This issue can be resolved by using information from the Hessian matrix to guide the search. The simplest method for doing so is known as **Newton’s method**. Newton’s method is based on using a second-order Taylor series expansion to approximate  `$f(x)$` near some point  `$x^{(0)}$`:

`$$
f(x) \approx f(x^{(0)}) +(x - x^{(0)})^{\mathrm{T}}\nabla_{x}f(x^{(0)}) + \frac{1}{2}(x - x^{(0)})^{\mathrm{T}}H(f)(x^{(0)})(x - x^{(0)}) \\
$$`

If we then solve for the critical point of this function, we obtain:

`$$
x^{*} = x^{(0)} - H(f)(x^{(0)})^{-1}\nabla_{x}f(x^{(0)}) \\
$$`

> 当 `$f$` 是一个正定二次函数时，牛顿法只要应用一次式就能直接跳到函数的最小点。如果 `$f$` 不是一个真正二次但能在局部近似为正定二次，牛顿法则需要多次迭代应用式。迭代地更新近似函数和跳到近似函数的最小点可以比梯度下降 **更快地** 到达临界点。这在接近局部极小点时是一个特别有用的性质，但是在鞍点附近 **是有害的**。当附近的临界点是最小点(Hessian 的所有特征值都是正的)时牛顿法才适用，而梯度下降不会被吸引到鞍点(除非梯度指向鞍点)。

Optimization algorithms that use only the gradient, such as gradient descent, are called **first-order optimization algorithms**. Optimization algorithms that also use the Hessian matrix, such as Newton’s method, are called **second-order optimization algorithms**.


## Constrained Optimization

Sometimes we wish not only to maximize or minimize a function `$f(x)$` over all possible values of `$x$`. Instead we may wish to find the maximal or minimal value of `$f(x)$` for values of `$s$` in some set `$\mathbb{S}$`. This is known as **constrained optimization**. Points `$x$` that lie within the set `$\mathbb{S}$` are called **feasible points** in constrained optimization terminology.

> 有时候,在 `$x$` 的所有可能值下最大化或最小化一个函数 `$f(x)$` 不是我们所希望的.相反,我们可能希望在 `$x$` 的某些集合 `$\mathbb{S}$` 中找 `$f(x)$` 的最大值或最小值.这被称为 **约束优化(constrained optimization)**.在约束优化术语中,集合 `$\mathbb{S}$` 内的点 `$x$` 被称为 **可行(feasible)点**.

We often wish to find a solution that is small in some sense. A common approach in such situations is to impose a norm constraint, such as `$\lVert x \rVert \leq 1$`.


The **Karush–Kuhn–Tucker (KKT)** approach provides a very general solution to constrained optimization. With the KKT approach, we introduce a new function called the **generalized Lagrangian** or **generalized Lagrange function**.

To define the Lagrangian, we first need to describe `$\mathbb{S}$` in terms of equations and inequalities. We want a description of `$\mathbb{S}$` in terms of m functions `$g^{(i)}$` and `$n$` functions `$h^{(j)}$` so that `$\mathbb{S} = \{x|\forall i,g^{(i)}(x)=0 \ \mathrm{and} \ \forall j,h^{(j)}(x) \leq 0 \}$`.Theequations involving `$g^{(i)}$` are called the **equality constraints** and the inequalities involving `$h^{(j)}$` are called **inequality constraints**.

We introduce new variables `$\lambda_{i}$` and `$\alpha_{j}$` for each constraint, these are called the KKT multipliers. The generalized Lagrangian is then defined as

`$$
L(x,\lambda,\alpha)=f(x)+ \sum_{i}\lambda_{i}g^{(i)}(x)+\sum_{j}\alpha_{j}h^{(j)}(x) \\
$$`


We can now solve a constrained minimization problem using unconstrained optimization of the generalized Lagrangian. Observe that, so long as at least one feasible point exists and `$f(x)$` is not permitted to have value `$\infty$`, then

`$$
\mathrm{min}_{x}\mathrm{max}_{\lambda}\mathrm{max}_{\alpha,\alpha \geq 0} L(x,\lambda,\alpha) \\
$$`

has the same optimal objective function value and set of optimal points `$x$` as

`$$
\mathrm{min}_{x \in \mathbb{S}} f(x) \\
$$`

This follows because any time the constraints are satisfied,

`$$
\mathrm{max}_{\lambda}\mathrm{max}_{\alpha,\alpha \geq 0} L(x,\lambda,\alpha) = f(x) \\
$$`

while any time a constraint is violated,

`$$
\mathrm{max}_{\lambda}\mathrm{max}_{\alpha,\alpha \geq 0} L(x,\lambda,\alpha) = \infty \\
$$`

These properties guarantee that no infeasible point can be optimal, and that the optimum within the feasible points is unchanged.

To perform constrained maximization, we can construct the generalized Lagrange function of `$-f(x)$`, which leads to this optimization problem:

`$$
\mathrm{min}_{x}\mathrm{max}_{\lambda}\mathrm{max}_{\alpha,\alpha \geq 0} -f(x) + \sum_{i}\lambda_{i}g^{(i)}(x)+\sum_{j}\alpha_{j}h^{(j)}(x)\\
$$`

We may also convert this to a problem with maximization in the outer loop:

`$$
\mathrm{max}_{x}\mathrm{min}_{\lambda}\mathrm{min}_{\alpha,\alpha \geq 0} -f(x) + \sum_{i}\lambda_{i}g^{(i)}(x)+\sum_{j}\alpha_{j}h^{(j)}(x) \\
$$`

The sign of the term for the equality constraints does not matter; we may define it with addition or subtraction as we wish, because the optimization is free to choose any sign for each `$\lambda_{i}$`.

A simple set of properties describe the optimal points of constrained opti- mization problems. These properties are called the Karush-Kuhn-Tucker (KKT) conditions. They are necessary conditions, but not always sufficient conditions, for a point to be optimal. The conditions are:

* The gradient of the generalized Lagrangian is zero.
* All constraints on both `$x$` and the KKT multipliers are satisfied.
* The inequality constraints exhibit "complementary slackness": `$\alpha \odot h(x) = 0$`

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/numerical.html.
