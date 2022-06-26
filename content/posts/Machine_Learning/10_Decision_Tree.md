---
title: "Decision Trees"
date: "2021-05-26"
tags: ["Machine Learning", "Decision Trees"]
categories: ["Data Science", "Machine Learning"]
weight: 3
---


A tree has many analogies in real life, and turns out that it has influenced a wide area of **machine learning**, covering both **classification** and **regression**. In decision analysis, a decision tree can be used to visually and explicitly represent decisions and decision making. As the name goes, it uses a tree-like model of decisions. Though a commonly used tool in data \mining for deriving a strategy to reach a particular goal, its also widely used in machine learning.

<div align="center">
<img src="/img_ML/10_What_is_tree.PNG" width=450px/>
</div>

**Decision Tree consists of :**

1. **Nodes** : Test for the value of a certain attribute.
2. **Edges/ Branch** : Correspond to the outcome of a test and connect to the next node or leaf.
3. **Leaf nodes** : Ter\minal nodes that predict the outcome (represent class labels or class distribution).

**Applications of Decision trees in real life** :

* Biomedical Engineering (decision trees for identifying features to be used in implantable devices).
* Financial analysis (Customer Satisfaction with a product or service).
* Astronomy (classify galaxies).
* System Control.
* Manufacturing and Production (Quality control, Semiconductor manufacturing, etc).
* Medicines (diagnosis, cardiology, psychiatry).
* Physics (Particle detection).

<br>

## Classification and Regression Trees (CART)


**Classification** and **regression** trees are machine-learning methods for constructing prediction models from data. The models are obtained by recursively partitioning the data space and fitting a simple prediction model within each partition. As a result, the partitioning can be represented graphically as a decision tree. Classification trees are designed for dependent variables that take a finite number of unordered values, with prediction error measured in terms of misclassification cost. Regression trees are for dependent variables that take continuous or ordered discrete values, with prediction error typically measured by the squared difference between the observed and predicted values.

<div align="center">
<img src="/img_ML/10_Regression_Classification.PNG" width=700px/>
</div>

Consider the data would be:

`$$
\{(x_{i},y_{i})\}_{i=1}^{n} \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
x = \begin{bmatrix}
x_{1},x_{2},...,x_{n}    \\
\end{bmatrix}_{\ d \times n}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
y = \begin{bmatrix}
y_{1}    \\
y_{2}    \\
...      \\
y_{n}    \\
\end{bmatrix}_{\ n \times 1}
$$`

Define:
* `$x_{j:}\to j'th$` row and all columns. (each row of `$x$` matrix is a feature)
* `$x_{:i}\to$` one column and all rows. (each column of `$x$` matrix is a data sample)

At each split we have two regions.

`$$
R_{1}(j,s) = \{ x_{:i}|x_{j:}\leq s \} \\
R_{2}(j,s) = \{ x_{:i}|x_{j:}> s \} \\
$$`

We need to decide about `$j$` and `$s$`. To decide about these parameters we can define an objective function.


### Regression Case

For simplicity, consider the model we want to fit for each region be a constant value. Therefore in region one for example, we want all the target values to be as close as possible to constant `$C_{1}$`.

`$$
\min_{C_{1}}\sum_{y_{i}\in R_{1}}(y_{i}-C_{1})^{2} \\
$$`

Since in each split we have two regions, we can write similar objective for region two.

`$$
\min_{C_{2}}\sum_{y_{i}\in R_{2}}(y_{i}-C_{2})^{2} \\
$$`

Therefore we have:

`$$
\min_{C_{1}}\sum_{y_{i}\in R_{1}}(y_{i}-C_{1})^{2}+\min_{C_{2}}\sum_{y_{i}\in R_{2}}(y_{i}-C_{2})^{2} \\
$$`

Finally we should also decide about `$j$` and `$s$`:

`$$
\min_{j,s}[\min_{C_{1}}\sum_{y_{i}\in R_{1}}(y_{i}-C_{1})^{2}+\min_{C_{2}}\sum_{y_{i}\in R_{2}}(y_{i}-C_{2})^{2}] \\
$$`

If `$j$` and `$s$` are known

`$$
C_{1} = \mathrm{Average}\{x_{:i}|x_{:i}\in R_{1} \} \\
C_{2} = \mathrm{Average}\{x_{:i}|x_{:i}\in R_{2} \} \\
$$`

Given `$j$`, we have only `$n$` possible choice for `$s$`

`$$
x_{j} = \begin{bmatrix}
x_{j1},x_{j2},...,x_{jn}    \\
\end{bmatrix}
$$`

One (trivial)solution for this problem can be a brute force search.
1. Consider feature `$j$` each time
2. Test all `$n$` possible choices for `$s$`
3. Find `$C_{1}$` and `$C_{2}$` for considered `$s$` and `$j$`
4. Repeat until finding \minimum value for objective function

Define

`$$
Q_{m}(T)=\frac{1}{n_{m}}\sum_{y_{i}\in R_{m}}(y_{i}-C_{m})^{2}
$$`

Where:
* `$m$` is a ter\minal node of the tree.
* `$R_{m}$` is a region.
* `$n_{m}$` is the number of points in `$R_{m}$`.
* `$C_{m}$` is the constant that we fit in `$R_{m}$`.
* |`$T$`| is the number of ter\minal nodes.

Therefore the **total error** would be:

`$$
\sum_{m=1}^{|T|}n_{m}Q_{m} \\
$$`

**Stopping criterion**

* **Option 1 (bad):** split only if the split reduces the residual sum of squares (RSS) by at least some threshold value
    * However, sometimes you have a split with a small improvement followed by one with a large improvement.
    * This stopping criterion would miss such splits
* **Option 2:** over‐build the tree and "prune" the tree later.
    * Pruning refers to removing some splits that create ter\minal leaves


### Tree Pruning

After a tree is fully extended, remove one leaf at a time to \minimize this criterion:

`$$
\sum_{m=1}^{|T|}n_{m}Q_{m} + \alpha|T|
$$`

Where

* `$\alpha|T|$` is the penalty for having too many leaves
* `$\alpha$` is a tuning parameter and can be chosen via cross validation
* Introducing the penalty is also called regularization

<div align="center">
<img src="/img_ML/10_Pruning.PNG" width=400px/>
</div>


### Classification Case

In regression case, we tried to fit a constant to a region. But in classification case we need to find the regions which contain samples from same class. We can change regression cost with some sort of misclassification cost.

* Consider the classes of
`$$C = \{1,2,3,...,𝑘\} \\ $$`
* Impurity:
`$$P_{mk}=\frac{1}{n_{m}}\sum_{y_{i}\in R_{m}}I(y_{i}=k) \\ $$`
* Misclassification:
`$$\frac{1}{n_{m}}\sum_{y_{i}\in R_{m}}I(y_{i}\neq k) = 1 - P_{mk} \\ $$`
* Gini Index:
`$$\sum_{i\neq j}P_{mi}P_{mj} \\ $$`
* Cross Entropy
`$$-\sum_{k=1}^{K}p_{mk}log(P_{mk}) \\ $$`

<div align="center">
<img src="/img_ML/10_Node_Impurity.PNG" width=500px/>
</div>

<br>

### Gini Example

In the snapshot below, we split the population using two input variables Gender and Class. Now, I want to identify which split is producing more homogeneous sub-nodes using Gini .

<div align="center">
<img src="/img_ML/10_Gini.PNG" width=800px/>
</div>

<br>

**Split on Gender:**
1. Calculate, Gini for sub-node `$\mathrm{Female} = (0.2)\times(0.2)+(0.8)\times(0.8)=0.68$`
2. Gini for sub-node `$\mathrm{Male} = (0.65)\times(0.65)+(0.35)\times(0.35)=0.55$`
3. Calculate weighted Gini for Split `$\mathrm{Gender} = (10/30)\times0.68+(20/30)\times0.55 = 0.59$`

**Similar for Split on Class:**
1. Gini for sub-node Class `$\mathrm{IX} = (0.43)\times(0.43)+(0.57)\times(0.57)=0.51$`
2. Gini for sub-node Class `$\mathrm{X} = (0.56)\times(0.56)+(0.44)\times(0.44)=0.51$`
3. Calculate weighted Gini for Split `$\mathrm{Class} = (14/30)\times0.51+(16/30)\times0.51 = 0.51$`

Above, we can see that Gini score for Split on Gender is higher than Split on Class, hence, the node split will take place on Gender.

<br>

##  Pros and Cons of Trees

### Pros:
* Inexpensive to construct.
* Extremely fast at classifying unknown records.
* Easy to interpret for small-sized trees
* Accuracy comparable to other classification techniques for many simple data sets.
* Excludes unimportant features.

### Cons:
* Decision Boundary restricted to being parallel to attribute axes.
* Decision tree models are often biased toward splits on features having a large number of levels.
* Small changes in the training data can result in large changes to decision logic.
* Large trees can be difficult to interpret and the decisions they make may seem counter intuitive.

<br>

## Reference

[1] Chakure, A. (2020, November 6). Decision Tree Classification. Medium. https://medium.com/swlh/decision-tree-classification-de64fc4d5aac.

[2] Płoński, P. (2020, June 22). Visualize a Decision Tree in 4 Ways with Scikit-Learn and Python. MLJAR Automated Machine Learning. https://mljar.com/blog/visualize-decision-tree/.
