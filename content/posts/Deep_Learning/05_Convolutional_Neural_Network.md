---
title: "Convolutional Neural Network"
date: "2021-08-02"
tags: ["CNN"]
categories: ["Deep Learning", "Data Science", "CNN"]
weight: 3
---

**Convolutional networks** (LeCun, 1989), also known as **convolutional neural networks** or CNNs, are a specialized kind of neural network for processing data that has a known, grid-like topology. Examples include time-series data, which can be thought of as a 1D grid taking samples at regular time intervals, and image data, which can be thought of as a 2D grid of pixels. Convolutional networks have been tremendously successful in practical applications. The name “convolutional neural network” indicates that the network employs a mathematical operation called **convolution**. Convolution is a specialized kind of linear operation. Convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in **at least one** of their layers.

> **卷积神经网络（Convolutional Neural Network, CNN** 是一种前馈神经网络，它的人工神经元可以响应一部分覆盖范围内的周围单元，对于大型图像处理有出色表现。它受到人类视觉神经系统的启发。卷积神经网络能够有效的将大数据量的图片降维成小数据量，同时能够有效的保留图片特征，符合图片处理的原则。

A typical layer of a convolutional network consists of three stages. In the first stage, the layer performs several **convolutions** in parallel to produce a set of linear activations.

In the second stage, each linear activation is run through a **nonlinear activation function**, such as the rectified linear activation function. This stage is sometimes called the **detector stage**.

In the third stage, we use a **pooling function** to modify the output of the layer further.

<div align="center">
  <img src="/img_DL/05_Convolution_Network_overview.PNG" width=500px/>
</div>
<br>

## The Convolution Operation

In its most general form, convolution is an operation on two functions of a real-valued argument.

`$$
s(t) = \int x(a)w(t-a)da \\
$$`

This operation is called **convolution**. The convolution operation is typically denoted with an asterisk:

`$$
s(t) = (x \ast w)(t) \\
$$`

> 对卷积这个名词的理解：**所谓两个函数的卷积，本质上就是先将一个函数翻转，然后进行滑动叠加**。先对`$w$`函数进行翻转，相当于在数轴上把`$w$`函数从右边褶到左边去，也就是卷积的 **“卷”** 的由来。然后再把`$w$`函数平移到 `$t$`，在这个位置对两个函数的对应点相乘，然后相加，这个过程是卷积的 **“积”** 的过程。

In convolutional network terminology, the first argument (in this example, the function `$x$`) to the convolution is often referred to as the **input** and the second argument (in this example, the function `$w$`) as the **kernel**. The output is sometimes referred to as the **feature map**.

 If we now assume that `$x$` and `$w$` are defined only on integer `$t$`, we can define the **discrete convolution**:

`$$
s[t] = (x \ast w)(t) = \sum_{a=-\infty}^{\infty} x[a]w[t-a] \\
$$`


> **Example: 信号分析**
>
> 输入信号是 `$f(t)$` ，是随时间变化的。系统响应函数是 `$g(t)$` ，图中的响应函数是随时间指数下降的，它的物理意义是说：如果在 `$t=0$` 的时刻有一个输入，那么随着时间的流逝，这个输入将不断衰减。换言之，到了 `$t=T$` 时刻，原来在 `$t=0$` 时刻的输入`$f(0)$`的值将衰减为`$f(0)g(T)$`。
>
><div align="center">
>  <img src="/img_DL/05_Convolution_Operation_example_p1.JPEG" width=500px/>
></div>
><br>
>
> 考虑到信号是连续输入的，也就是说，每个时刻都有新的信号进来，所以，最终输出的是所有之前输入信号的累积效果。如下图所示，在`$T=10$`时刻，输出结果跟图中带标记的区域整体有关。其中，`$f(10)$`因为是刚输入的，所以其输出结果应该是`$f(10)g(0)$`，而时刻`$t=9$`的输入`$f(9)$`，只经过了1个时间单位的衰减，所以产生的输出应该是 `$f(9)g(1)$`，如此类推，即图中虚线所描述的关系。这些对应点相乘然后累加，就是`$T=10$`时刻的输出信号值，这个结果也是`$f$`和`$g$`两个函数在`$T=10$`时刻的卷积值。
>
><div align="center">
>  <img src="/img_DL/05_Convolution_Operation_example_p2.JPEG" width=500px/>
></div>
><br>
>
> ​​显然，上面的对应关系看上去比较难看，是拧着的，所以，我们把`$g$`函数对折一下，变成了`$g(-t)$`，这样就好看一些了。看到了吗？这就是为什么卷积要“卷”，要翻转的原因，这是从它的物理意义中给出的。
>
><div align="center">
>  <img src="/img_DL/05_Convolution_Operation_example_p3.JPEG" width=500px/>
></div>
><br>
>
> ​上图虽然没有拧着，已经顺过来了，但看上去还有点错位，所以再进一步平移`$T$`个单位，就是下图。它就是本文开始给出的卷积定义的一种图形的表述：
>
><div align="center">
>  <img src="/img_DL/05_Convolution_Operation_example_p4.JPEG" width=600px/>
></div>
><br>
>
> ​​所以，在以上计算`$T$`时刻的卷积时，要维持的约束就是： `$t+(T-t)=T$` 。

We often use convolutions over more than one axis at a time. For example, if we use a two-dimensional image `$I$` as our input, we probably also want to use a two-dimensional kernel `$K$`:

`$$
s[i,j] = (I * K)[i,j] = \sum_{m}\sum_{n}I[m,n]K[i-m,j-n] \\
$$`

Convolution is **commutative**, meaning we can equivalently write:

`$$
s[i,j] = (K * I)[i,j] = \sum_{m}\sum_{n}I[i-m,j-n]K[m,n] \\
$$`


The commutative property of convolution arises because we have **flipped** the kernel relative to the input, in the sense that as m increases, the index into the input increases, but the index into the kernel decreases. The only reason to flip the kernel is to obtain the commutative property. While the commutative property is useful for writing proofs, it is not usually an important property of a neural network implementation. Instead, many neural network libraries implement a related function called the cross-correlation, which is the same as convolution but without flipping the kernel:

`$$
s[i,j] = (I * K)[i,j] = \sum_{m}\sum_{n}I[i-m,j-n]K[m,n] \\
$$`

Discrete convolution can be viewed as multiplication by a matrix. However, the matrix has several entries constrained to be equal to other entries. For example, for univariate discrete convolution, each row of the matrix is constrained to be equal to the row above shifted by one element. This is known as a **Toeplitz matrix**.

<div align="center">
  <img src="/img_DL/05_Convolution_Operation_exp.PNG" width=500px/>
</div>
<br>

> 这个过程我们可以理解为我们使用一个 **过滤器（卷积核)** 来过滤图像的各个小区域，从而得到这些小区域的特征值。

<div align="center">
  <img src="/img_DL/05_Convolution_Operation.gif" width=400px/>
</div>
<br>

> 在具体应用中，往往有多个卷积核，可以认为，每个卷积核代表了一种 **图像模式** ，如果某个图像块与此卷积核卷积出的值大，则认为此图像块十分接近于此卷积核。如果我们设计了6个卷积核，可以理解：我们认为这个图像上有6种底层纹理模式，也就是我们用6中基础模式就能描绘出一副图像。以下就是25种不同的卷积核的示例：

<div align="center">
  <img src="/img_DL/05_Convolution_kernel_example.JPEG" width=200px/>
</div>
<br>

## Motivation

Convolution leverages three important ideas that can help improve a machine learning system: **sparse interactions**, **parameter sharing** and **equivariant representations**. Moreover, convolution provides a means for working with inputs of variable size.

Traditional neural network layers use matrix multiplication by a matrix of parameters with a separate parameter describing the interaction between each input unit and each output unit. This means every output unit interacts with every input unit. Convolutional networks, however, typically have **sparse interactions** (also referred to as **sparse connectivity** or **sparse weights**). This is accomplished by making the kernel smaller than the input.

> 卷积网络具有 **稀疏交互(sparse interactions)** (也叫做 **稀疏连接(sparse connectivity)** 或者 **稀疏权重(sparse weights))** 的特征。这是使核的大小远小于输入的大小来达到的。当处理一张图像时，输入的图像可能包含成千上万个像素点，但是我们可以通过只 占用几十到上百个像素点的核来检测一些小的有意义的特征，例如图像的边缘。这意味着我们需要存储的参数更少，不仅减少了模型的存储需求，而且提高了它的统计效率。这也意味着为了得到输出我们只需要更少的计算量。这些效率上的提高往往是很显著的。

**Parameter sharing** refers to using the same parameter for more than one function in a model. In a traditional neural net, each element of the weight matrix is used exactly once when computing the output of a layer. It is multiplied by one element of the input and then never revisited. In a convolutional neural net, each member of the kernel is used at every position of the input (except perhaps some of the boundary pixels, depending on the design decisions regarding the boundary). The parameter sharing used by the convolution operation means that rather than learning a separate set of parameters for every location, we learn only one set. This does not affect the runtime of forward propagation, but it does further reduce the storage requirements of the model to `$k$` parameters.

> **参数共享(parameter sharing)** 是指在一个模型的多个函数中使用相同的参数。在卷积神经网络中，核的每一个元素都作用在输入的每一位置上。卷积运算中的参数共享保证了我们只需要学习一个参数集合，而不是对于每一位置都需要学习一个单独的参数集合。

In the case of convolution, the particular form of parameter sharing causes the layer to have a property called **equivariance** to translation. To say a function is equivariant means that if the input changes, the output changes in the same way. Specifically, a function `$f(x)$` is equivariant to a function `$g$` if

`$$
f(g(x)) = g(f(x)) \\
$$`

> 对于卷积，参数共享的特殊形式使得神经网络层具有对平移 **等变(equivariance)** 的性质。如果一个函数满足输入改变，输出也以同样的方式改变这一性质，我们就说它是等变 (equivariant) 的。

A convolutional layer have equivariance to translation. For example

`$$
g(x)[i] = x[i-1] \\
$$`

If we apply this transformation to `$x$`, then apply convolution, the result will be the same as if we applied convolution to `$x$`, then applied the transformation to the output.

For images, convolution creates a 2-D map of where certain features appear in the input. Note that convolution is not equivariant to some other transformations, such as changes in the scale or rotation of an image.

## Pooling

A pooling function replaces the output of the net at a certain location with a summary statistic of the nearby outputs. For example, the **max pooling** (Zhou and Chellappa, 1988) operation reports the maximum output within a rectangular neighborhood. Other popular pooling functions include the average of a rectangular neighborhood, the `$L^{2}$` norm of a rectangular neighborhood, or a weighted average based on the distance from the central pixel.

<div align="center">
  <img src="/img_DL/05_Max_Pooling.PNG" width=600px/>
</div>
<br>

In all cases, pooling helps to make the representation become approximately invariant to small translations of the input. Invariance to translation means that if we translate the input by a small amount, the values of most of the pooled outputs do not change. **Invariance to local translation can be a very useful property if we care more about whether some feature is present than exactly where it is**.

> **池化层** 是一个利用 **池化函数 (pooling function)** 对网络输出进行进一步调整的网络层。池化函数使用某一位置的相邻输出的总体统计特征来代替网络在该位置的输出。常用的池化函数包括最大池化 (max pooling) 函数 (即给出邻域内的最大值) 和平均池化 (average pooling) 函数 (即给出邻域内的平均值) 等。但无论选择何种池化函数，当对输入做出少量平移时，池化对输入的表示都近似 **不变 (invariant)**。局部平移不变性 是一个很重要的性质，尤其是当我们关心某个特征是否出现而不关心它出现的位置时。

The use of pooling can be viewed as adding an infinitely strong prior that the function the layer learns must be invariant to small translations. When this assumption is correct, it can greatly improve the statistical efficiency of the network.

Pooling over spatial regions produces invariance to translation, but if we pool over the outputs of separately parametrized convolutions, the features can learn
which transformations to become invariant to.

It is also possible to dynamically pool features together, for example, by running a clustering algorithm on the locations of interesting features (Boureau et al., 2011). This approach yields a different set of pooling regions for each image. Another approach is to learn a single pooling structure that is then applied to all images (Jia et al., 2012).

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/convnets.html.
