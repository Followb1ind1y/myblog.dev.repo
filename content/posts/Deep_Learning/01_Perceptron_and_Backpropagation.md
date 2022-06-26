---
title: "Neural Network: Perceptron and Backpropagation"
date: "2021-07-03"
tags: ["Perceptron", "Feedforward Deep Networks", "Backpropagation"]
categories: ["Data Science", "Deep Learning", "Neural Network"]
weight: 3
---


**Neural Networks** form the base of deep learning, which is a subfield of Machine Learning, where the structure of the human brain inspires the algorithms. Neural networks take input data, train themselves to recognize patterns found in the data, and then predict the output for a new set of similar data. Therefore, a neural network can be thought of as the functional unit of deep learning, which mimics the behavior of the human brain to solve complex data-driven problems.

<div align="center">
  <img src="/img_DL/01_AI_ML_DL.PNG" width=400px/>
</div>

The first thing that comes to our mind when we think of "neural networks" is biology, and indeed, neural nets are inspired by our brains. In machine learning, the neurons' **dendrites** refer to as input, and the nucleus process the data and forward the calculated output through the **axon**. In a biological neural network, the width (thickness) of dendrites defines the weight associated with it.

<div align="center">
  <img src="/img_DL/01_biological_neuron.jpeg" width=700px/>
</div>


## Perceptron

A **perceptron** is a neural network without any hidden layer. A perceptron only has an input layer and an output layer. Perceptrons can be represented graphically as:

<div align="center">
  <img src="/img_DL/01_Perceptron.PNG" width=600px/>
</div>

Where `$x_{i}$` is the `$i$`-th feature of a sample and `$\beta_{i}$` is the `$i$`-th weight. `$\beta_{0}$` is defined as the bias. The bias alters the position of the decision boundary between the two classes. From a geometrical point of view, Perceptron assigns label "1" to elements on one side of `$\beta^{T}x+\beta_{0}$` and label "-1" to elements on the other side. Define a cost function, `$\phi(\beta,\beta_{0})$`, as a summation of the distance between all misclassified points and the hyperplane, or the decision boundary. To minimize the cost function, we need to estimate `$\beta$`, `$\beta_{0}$`.

`$$
\min_{\beta,\beta_{0}}\phi(\beta,\beta_{0})=\mathrm{distance \ of \ all \ misclassified \ points} \\
$$`

<div align="center">
  <img src="/img_DL/01_decision_boundary.PNG" width=250px/>
</div>

**1.** A hyperplane `$L$` can be defined as:

`$$
L=\{x:f(x)= \beta^{T}x+\beta_{0}=0\} \\
$$`

For any arbitrary points `$x_{1}$` and `$x_{2}$` on `$L$`, we have

`$$
\beta^{T}x_{1}+\beta_{0}=0 \\
\beta^{T}x_{2}+\beta_{0}=0 \\
\mathrm{Such \ that \ } \beta^{T}(x_{1}-x_{2})=0 \\
$$`

**2.** For any `$x_{0}$` on the hyperplane,

`$$
\beta^{T}x_{0}+\beta_{0}=0 \Rightarrow \beta^{T}x_{0} = -\beta_{0} \\
$$`

**3.** We set `$\beta^{*}=\frac{\beta}{\parallel\beta\parallel}$` as the unit normal vector of the hyperplane `$L$`. For simplicity we can call `$\beta^{*}$` norm vector. The distance of point `$x$` to `$L$` is given by

`$$
\beta^{*T}(x-x_{0}) = \beta^{*T}x - \beta^{*T}x_{0}= \frac{\beta^{T}x}{\parallel\beta\parallel}+\frac{\beta_{0}}{\parallel\beta\parallel} = \frac{(\beta^{T}x+\beta_{0})}{\parallel\beta\parallel} \\
$$`

Where `$x_{0}$` is any point on `$L$`. Hence, `$\beta^{T}x+\beta_{0}$` is proportional to the distance of the point `$x$` to the hyperplane `$L$`.

**4.** The distance from a misclassified data point `$x_{i}$` to the hyperplane `$L$` is

`$$
d_{i}=-y_{i}(\beta^{T}x_{i}+\beta_{0}) \\
$$`

Where `$y_{i}$` is a target value, such that `$y_{i}=1$` if `$\beta^{T}x_{i}+\beta_{0}<0$`, `$y_{i}=-1$` if `$\beta^{T}x_{i}+\beta_{0}>0$`

Since we need to find the distance from the hyperplane to the misclassified data points, we need to add a negative sign in front. When the data point is misclassified, `$\beta^{T}x_{i}+\beta_{0}$` will produce an opposite sign of `$y_{i}$`. Since we need a positive sign for distance, we add a negative sign.

### Perceptron Learning using Gradient Descent

The gradient descent is an optimization method that finds the minimum of an objective function by incrementally updating its parameters in the negative direction of the derivative of this function. In our case, the objective function to be minimized is classification error and the parameters of this function are the weights associated with the inputs `$\beta$`. The gradient descent algorithm updates the weights as follows:

`$$
\beta^{new} \leftarrow \beta^{old} - \rho \frac{\partial Err}{\partial \beta} \\
$$`

Where `$\rho$` is called the learning rate.

The classification error can be defined as the distance of misclassified observations to the decision boundary,

`$$
D(\beta) = -\sum_{i\in M}y_{i}\beta^{T}x_{i} \\
$$`

Where `$M$` is the set of misclassified points. The quantity `$y_{i}\beta^{T}x_{i}$` will be negative if `$x_{i}$`
is misclassified. By taking the derivative of `$D(\beta)$` with respect to `$\beta$`

`$$
\begin{align*}
\frac{\partial D}{\partial \beta} &= - \sum_{i\in M}y_{i}x_{i} \\
\frac{\partial D}{\partial \beta_{0}} &= - \sum_{i\in M}y_{i} \\
\end{align*}
$$`

The update formula becomes

`$$
\beta^{new} \leftarrow \beta^{old} + \rho \sum_{i\in M}y_{i}x_{i} \\
$$`

Which is equivalent to incrementally updating `$\beta$` for each misclassified point `$x_{i}$`

`$$
\beta^{new} \leftarrow \beta^{old} + \rho y_{i}x_{i} \\
$$`

The intuition behind this update is that for misclassified point `$x_{i}$`, `$\beta$` should be changed in the direction that makes `$x_{i}$` as close as possible to the right side. Figure 2 shows how `$\beta$` is updated.

<div align="center">
  <img src="/img_DL/01_GD.PNG" width=350px/>
</div>

### Separability and Convergence

The training set `$D$` is said to be linearly separable if there exits a positive constant `$\gamma$` and a weight vector `$\beta$` such that `$(\beta^{T}x_{i}+\beta_{0})y_{i}>\gamma$` for all `$1<i<n$`. That is, if we say that `$\beta$` is the weight vector of Perceptron and `$y_{i}$` is the true label of `$x_{i}$`, then the signd distance of the `$x_{i}$` from `$\beta$` is greater than a positive constant `$\gamma$` for any `$(x_{i},y_{i})\in D$`.

If data is linearly-separable, the solution is theoretically guranteed to converge to a separating hyperplane in a finite numver of iterations. In this situation the number of iterations depends on the learning rate and the margin. However, if the data is not linearly separable there is no guarantee that the algorithm converges.

### Features

* A Perceptron can only discriminate between two classes at a time.

* When data is (linearly) separable, there are an infinite number of solutions depending on the starting point.

* Even though convergence to a solution is guaranteed if the solution exists, the finite number of steps until convergence can be very large.

* The smaller the gap between the two classes, the longer the time of convergence.

* When the data is not separable, the algorithm will not converge (it should be stopped after N steps).

* A learning rate that is too high will make the perceptron periodically oscillate around the solution unless additional steps are taken.

* The L.S. compute a linear combination of feature of input and return the sign.

* Learning rate affects the accuracy of the solution and the number of iterations directly.

## Feedforward Deep Networks

Feedforward neural networks are artificial neural networks where the connections between units do not form a cycle. Feedforward neural networks were the first type of artificial neural network invented and are simpler than their counterpart, recurrent neural networks. They are called feedforward because information only travels forward in the network (no loops), first through the input nodes, then through the hidden nodes (if present), and finally through the output nodes.

Feedforward neural network is a multistage regression or classification model typically represented by a graphical diagram. **Regression** usually produces **one** output unit `$Y_{1}$` while for `$k$` - classification there are `$k$` output units `$Y_{1...k}$` with each `$Y_{k}$` coded as 0 − 1 to represent the `$k^{th}$` class.

<div align="center">
  <img src="/img_DL/01_FNN.PNG" width=600px/>
</div>

where `$a_{i} = u \cdot x$` and `$z_{i} = \phi(a_{i})$` which is a non-linear function with an example being `$\phi(a) = \frac{1}{1+e^{−a}}$`. The function `$\phi$` is called the activation function and is used in classification not regression.

Feedforward deep networks, a.k.a. multilayer perceptrons (MLPs), are parametric function composed of several parametric function. Each layer of the network defines one of these sub-functions. Each layer (sub-function) has multiple inputs and multiple outputs. Each layer composed of many units (scalar output of the layer). We sometimes refer to each unit as a feature. Each unit is usually a simple transformation of its input. Also, the entire network can be very complex.

> **深度前馈网络(deep feedforward network)**，也叫作 **前馈神经网络(feedforward neural network)** 或者 **多层感知机(multilayer perceptron, MLP)** ，是典型的深度学习模型。前馈网络的目标是近似某个函数 `$f^{*}$`。

##  Backpropagation

Back-propagation is the essence of neural net training. It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch (i.e., iteration). Proper tuning of the weights allows you to reduce error rates and to make the model reliable by increasing its generalization.

Backpropagation is a short form for "backward propagation of errors." It is a standard method of training artificial neural networks. This method helps to calculate the gradient of a loss function with respects to all the weights in the network.

<div align="center">
  <img src="/img_DL/01_backprob.PNG" width=650px/>
</div>

`$$
Error = |\hat{Y}-Y|^{2}, \ \ \ \ \ \ \ \ \ \ a_{i} = \sum_{l}z_{l}u_{il}, \ \ \ \ \ \ \ \ \ \ z_{i} = \sigma(a_{i}), \ \ \ \ \ \ \ \ \ \ \sigma(a) = \frac{1}{1+e^{-a}}
$$`

Take the derivative with respect to weight `$u_{il}$`

`$$
\frac{\partial Error}{\partial u_{il}} = \underbrace{\frac{\partial Error}{\partial a_{i}}}_{\delta_{i}\ (Unknown)} \cdot \underbrace{\frac{\partial a_{i}}{\partial u_{il}}}_{z_{l} \ (Known)} \\
$$`

`$$
\begin{align*}
\delta_{i} = \frac{\partial Error}{\partial a_{i}} &= \sum_{j}\underbrace{\frac{\partial Error}{\partial a_{j}}}_{\delta_{j}} \cdot \frac{\partial a_{j}}{\partial a_{i}} \to (\frac{\partial a_{j}}{\partial z_{i}}\cdot \frac{\partial z_{i}}{\partial a_{i}}) \\
&= \sum_{j}\delta_{j} \cdot u_{ji} \cdot \sigma'(a_{i}) = \sigma'(a_{i})\sum_{j}\delta_{j} \cdot u_{ji}
\end{align*}
$$`

Note that if `$\sigma(x)$` is the sigmoid function, then

`$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$`

Now considering `$\delta_{k}$` for the output layer:

`$$
\delta_{k} = \frac{\partial Error}{\partial a_{k}} = \frac{\partial (y-\hat{y})^{2}}{\partial a_{k}} = -2(y-\hat{y})  \ \ \ \ \ \ \ \ \ \ where \ a_{k} = \hat{y}
$$`

The network weights are updated using the backpropagation algorithm when each training data point `$x$` is fed into the feed forward neural network (FFNN).

`$$
u_{il}^{new} \leftarrow u_{il}^{old} - \rho \cdot \frac{\partial (y-\hat{y})^{2}}{\partial u_{il}}
$$`


### Backpropagation procedure


1. First arbitrarily choose some random weights (preferably close to zero) for your network.


2. Apply `$x$` to the FFNN's input layer, and calculate the outputs of all input neurons.


3. Propagate the outputs of each hidden layer forward, one hidden layer at a time, and calculate the outputs of all hidden neurons.


4. Once `$x$` reaches the output layer, calculate the output(s) of all output neuron(s) given the outputs of the previous hidden layer.


5. At the output layer, compute `$\delta_{k} = −2(y_{k} − \hat{y}_{k} )$` for each output neuron(s).


6. Compute each `$\delta_{i}$`, starting from `$i = k − 1$` all the way to the first hidden layer, where `$\delta_{i}=\sigma'(a_{i})\sum_{j}\delta_{j} \cdot u_{ji}$`


7. Compute `$\frac{\partial (y-\hat{y})^{2}}{\partial u_{il}} = \delta_{i}z_{l}$` for all weights `$u_{il}$`.


8. Then update `$u_{il}^{new} \leftarrow u_{il}^{old} - \rho\cdot \frac{\partial (y-\hat{y})^{2}}{\partial u_{il}}$` for all weights `$u_{il}$`.


9. Continue for next data points and iterate on the training set until weights converge.

It is common to cycle through the all of the data points multiple times in order to reach convergence. An epoch represents one cycle in which you feed all of your datapoints through the neural network. It is good practice to randomized the order you feed the points to the neural network within each epoch; this can prevent your weights changing in cycles. The number of epochs required for convergence depends greatly on the learning rate & convergence requirements used.


## Reference

[1] Odegua, R. (2021, April 8). Building a Neural Network From Scratch Using Python (Part 1). Medium. https://heartbeat.fritz.ai/building-a-neural-network-from-scratch-using-python-part-1-6d399df8d432.

[2]  Shukla, P., &amp; Iriondo, R. (2021, April 2). Neural Networks from Scratch with Python Code and Math in Detail- I. Medium. https://pub.towardsai.net/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf#3a44.
