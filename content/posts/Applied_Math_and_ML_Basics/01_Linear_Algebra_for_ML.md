---
title: "[ML Basics] Linear Algebra"
date: "2021-06-23"
tags: ["Linear Algebra"]
categories: ["Machine Learning Basics", "Data Science"]
weight: 3
---

## Scalars, Vectors, Matrices and Tensors

The study of linear algebra involves several types of mathematical objects:

* **Scalars**: A scalar is just a single number, in contrast to most of the other objects studied in linear algebra, which are usually arrays of multiple numbers. We write scalars in italics. We usually give scalars lowercase variable names. When we introduce them, we specify what kind of number they are. For example, we might say "Let `$s \in \mathbb{R}$` be the slope of the line," while defining a real-valued scalar, or "Let `$n \in \mathbb{N}$` be the number of units, while defining an natural number scalar.

* **Vectors**: A vector is an array of numbers. The numbers are arranged in order. We can identify each individual number by its index in that ordering. Typically we give vectors lowercase names in bold typeface, such as `$\mathbf{x}$`. The elements of the vector are identified by writing its name in italic typeface, with a subscript. We also need to say what kind of numbers are stored in the vector. If each element is in `$\mathbb{R}$`, and the vector has `$n$` elements, then the vector lies in the set formed by taking the Cartesian product of `$\mathbb{R}$` `$n$` times, denoted as `$\mathbb{R}^{n}$`. When we need to explicitly identify the elements of a vector, we write them as a column enclosed in square brackets:

`$$
\begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n} \\
\end{bmatrix}
$$`

* **Matrices**: A matrix is a 2-D array of numbers, so each element is identified by two indices instead of just one. We usually give matrices upper-case variable names with bold typeface, such as `$\mathbf{A}$`. If a real-valued matrix `$\mathbf{A}$` has a height of `$m$` and a width of `$n$`, then we say that `$\mathbf{A} \in \mathbb{R}^{m \times  n}$`. We usually identify the elements of a matrix using its name in italic but not bold font, and the indices are listed with separating commas. For example, `$A_{1,1}$` is the upper left entry of `$\mathbf{A}$` and `$A_{m,n}$` is the bottom right entry. We can identify all of the numbers with vertical coordinate `$i$` by writing a ":" for the horizontal coordinate. For example, `$A_{i,:}$` denotes the horizontal cross section of `$\mathbf{A}$` with vertical coordinate `$i$`. This is known as the `$i$`-th **row** of `$\mathbf{A}$`. Likewise, `$A_{:,i}$` is the `$i$`-th **column** of `$\mathbf{A}$`. When we need to explicitly identify the elements of a matrix, we write them as an array enclosed in square brackets:

`$$
\begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2,2} \\
\end{bmatrix}
$$`

* **Tensors**: In some cases we will need an array with more than two axes. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor. We denote a tensor named "A" with this typeface: `$\mathsf{A}$`. We identify the element of `$\mathsf{A}$` at coordinates `$(i,j,k)$` by writing `$A_{i,j,k}$`.

> **标量(Scalar)**: 标量是一个单独的数. <br>
**向量(Vector)**: 向量是一列有序排列的数. <br>
**矩阵(Matrix)**: 矩阵是一个二维数组, 其中的每一个元素被两个索引. <br>
**张量(Tensor)**: 张量是一个超过二维的数组. <br>

One important operation on matrices is the **transpose**. The transpose of a matrix is the mirror image of the matrix across a diagonal line, called the **main diagonal**, running down and to the right, starting from its upper left corner. We denote the transpose of a matrix `$A$` as `$A^{T}$`, and it is defined such that

`$$
(A^{T})_{i,j} = A_{j,i}
$$`

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_01_Transpose.PNG" width=500px/>
</div>
<br>

Vectors can be thought of as matrices that contain only one column. The transpose of a vector is therefore a matrix with only one row. Sometimes we define a vector by writing out its elements in the text inline as a row matrix, then using the transpose operator to turn it into a standard column vector, e.g., `$x = \begin{bmatrix}
x_{1}, x_{2}, x_{3} \\
\end{bmatrix}^{T}$`.

A scalar can be thought of as a matrix with only a single entry. From this, we can see that a scalar is its own transpose: `$a = a^{T}$`.

> **转置(Transpose)** 是矩阵的一种重要操作, 矩阵的转置是以对角线为轴的镜像, 这条从左上角到右下角的对角线被称为**主对角线**. 换句话来说, 矩阵的转置可以看成以主对角线为轴的一个镜像. <br>
**向量**可以看作只有一列的矩阵。对应地, 向量的转置可以看作是只有一行的矩阵.<br>
**标量**可以看作是只有一个元素的矩阵. 因此, 标量的转置等于它本身. <br>

We can add matrices to each other, as long as they have the same shape, just by adding their corresponding elements: `$C = A + B$` where `$C_{i,j} = A_{i,j} + B_ {i,j}$`.

We can also add a scalar to a matrix or multiply a matrix by a scalar, just by performing that operation on each element of a matrix: `$D = a \cdot B + c$` where `$D_{i,j} =a \cdot B_{i,j} +c$`.

In the context of deep learning, we also use some less conventional notation. We allow the addition of matrix and a vector, yielding another matrix: `$C = A + b$`, where `$C_{i,j} = A_{i,j} + b_{j}$` . In other words, the vector `$b$` is added to each row of the matrix. This shorthand eliminates the need to define a matrix with `$b$` copied into each row before doing the addition. This implicit copying of `$b$` to many locations is called **broadcasting**.

> 只要矩阵的**形状一样**, 我们可以把**两个矩阵**相加是指对应位置的元素相加. <br>
**标量和矩阵**相乘, 或是和矩阵相加时, 我们只需将其与矩阵的每个元素相乘或相加. <br>

>深度学习中, 我们允许**矩阵和向量相加**. 方法是将向量和矩阵的每一行相加. 这个简写方法使我们无需在加法操作前定义一个将向量复制到每一行而生成的矩阵. 这种隐式地复制向量到很多位置的方式, 被称为**广播(Broadcasting)**.

## Multiplying Matrices and Vectors

One of the most important operations involving matrices is multiplication of two matrices. The **matrix product** of matrices `$A$` and `$B$` is a third matrix `$C$`. In order for this product to be defined, `$A$` must have the same number of columns as `$B$` has rows. If `$A$` is of shape `$m \times n$` and `$B$` is of shape `$n \times p$`, then `$C$` is of shape `$m \times p$`. We can write the matrix product just by placing two or more matrices together, e.g.

`$$
C_{m \times p} = A_{m \times n}B_{n \times p} \\
$$`

The product operation is defined by

`$$
C_{i,j} = \sum_{k}A_{i,k}B_{k,j} \\
$$`

For example, if `$A = \begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2,2} \\
A_{3,1} & A_{3,2} \\
\end{bmatrix}$` and `$B = \begin{bmatrix}
B_{1,1} & B_{1,2} \\
B_{2,1} & B_{2,2} \\
\end{bmatrix}$`, then

`$$
C = A \times B = \begin{bmatrix}
A_{1,1}B_{1,1} + A_{1,2}B_{2,1} & A_{1,1}B_{1,2} + A_{1,2}B_{2,2} \\
A_{2,1}B_{1,1} + A_{2,2}B_{2,1} & A_{2,1}B_{1,2} + A_{2,2}B_{2,2} \\
A_{3,1}B_{1,1} + A_{3,2}B_{2,1} & A_{3,1}B_{1,2} + A_{3,2}B_{2,2} \\
\end{bmatrix}
$$`

> 两个**矩阵的乘法**仅当第一个矩阵 `$A$` 的**列数(column)** 和另一个矩阵 `$B$` 的**行数(row)** 相等时才能定义. 如 `$A$` 是一个 `$m \times n$` 的矩阵, `$B$` 是一个 `$n \times p$` 的矩阵，那它们的乘积 `$AB$` 就会是一个 `$m \times p$` 的矩阵.

Note that the standard product of two matrices is not just a matrix containing the product of the individual elements. Such an operation exists and is called the **element-wise** product or **Hadamard product**, and is denoted as `$A \odot B$`.

For example,

`$$
\begin{bmatrix}
A_{1,1} & A_{1,2} \\
A_{2,1} & A_{2,2} \\
\end{bmatrix} \odot \begin{bmatrix}
B_{1,1} & B_{1,2} \\
B_{2,1} & B_{2,2} \\
\end{bmatrix} = \begin{bmatrix}
A_{1,1} \cdot B_{1,1} & A_{1,2} \cdot B_{1,2} \\
A_{2,1} \cdot B_{2,1} & A_{2,2} \cdot B_{2,2} \\
\end{bmatrix} \\
$$`

> 两个矩阵中对应元素的乘积被称为**元素对应乘积(element-wise product)** 或者**Hadamard乘积(Hadamard product)**, 记为 `$A \odot B$`.

The **dot product** between two vectors `$x$` and `$y$` of the same dimensionality is the matrix product `$x^{T}y$`. We can think of the matrix product `$C=AB$` as computing `$C_{i,j}$` as the dot product between row `$i$` of `$A$` and column `$j$` of `$B$`.

Matrix product operations have many useful properties that make mathematical analysis of matrices more convenient. For example, matrix multiplication is distributive:

`$$
A(B+C)=AB+AC \\
$$`

It is also associative:

`$$
A(BC)=(AB)C \\
$$`

> 矩阵的乘法满足结合律和对矩阵加法的分配律:
>* **结合律**：`$A(BC)=(AB)C$` <br>
>* **分配律**：`$A(B+C)=AB+AC$`<br>

Matrix multiplication is **not** commutative (the condition `$AB = BA$` does **not** always hold), unlike scalar multiplication. However, the dot product between two vectors is commutative:

`$$
x^{T}y = y^{T}x \\
$$`

The transpose of a matrix product has a simple form:

`$$
(AB)^{T} = B^{T}A^{T} \\
$$`

> 矩阵的乘法与数乘运算之间也满足类似结合律的规律；与转置之间则满足倒置的分配律:
> * `$c(AB)=(cA)B=A(cB)$` <br>
> * `$(AB)^{T} = B^{T}A^{T}$` <br>

Now, We can write down a **system of linear equations**:

`$$
Ax = b \\
$$`

Where `$A \in R^{m \times n}$` is a known matrix, `$b \in R^{m}$` is a known vector, and `$x \in R^{n}$` is a vector of unknown variables we would like to solve for. Each element `$x_{i}$` of `$x$` is one of these unknown variables. Each row of `$A$` and each element of `$b$` provide another constraint.

`$$
Ax = b \Rightarrow \begin{bmatrix}
A_{1,1} & A_{1,2} & \cdots & A_{1,n}\\
A_{2,1} & A_{2,2} & \cdots & A_{2,n}\\
\vdots & \vdots & \ddots & \vdots\\
A_{m,1} & A_{m,2} & \cdots & A_{m,n}\\
\end{bmatrix} \times \begin{bmatrix}
x_{1} \\
x_{2} \\
\vdots \\
x_{n} \\
\end{bmatrix} = \begin{bmatrix}
b_{1} \\
b_{2} \\
\vdots \\
b_{m} \\
\end{bmatrix} \\
$$`

`$$
\Rightarrow \begin{cases}
A_{1,1}x_{1}+A_{1,2}x_{2}+ \cdots +A_{1,n}x_{n} = b_{1} \\
A_{2,1}x_{1}+A_{2,2}x_{2}+ \cdots +A_{2,n}x_{n} = b_{2} \\
\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \cdots \\
A_{m,1}x_{1}+A_{m,2}x_{2}+ \cdots +A_{m,n}x_{n} = b_{m} \\
\end{cases}
$$`

##  Identity and Inverse Matrices

Linear algebra offers a powerful tool called **matrix inversion** that allows us to
analytically solve equation `$Ax=b$` for many values of `$A$`.

To describe matrix inversion, we first need to define the concept of an **identity matrix**. An identity matrix is a matrix that does not change any vector when we multiply that vector by that matrix. We denote the identity matrix that preserves `$n$`-dimensional vectors as `$I_{n}$`. Formally, `$I_{n} \in \mathbb{R}^{n \times n}$`, and

`$$
\forall x \in \mathbb{R}^{n}, I_{n}x=x \\
$$`

The structure of the identity matrix is simple: all of the entries along the main diagonal are 1, while all of the other entries are zero. For example,

`$$
I_{3} = \begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1 \\
\end{bmatrix}
$$`

> **单位矩阵(identity matrix)** 的概念为, 任意向量和单位矩阵相乘, 都不会改变. 我们将保持`$n$`维向量不变的单位矩阵记作 `$I_{n}$`. 单位矩阵的结构很简单: 所有沿主对角线的元素都是1, 而所有其他位置的元素都是0.

The **matrix inverse** of `$A$` is denoted as `$A^{-1}$`, and it is defined as the matrix such that

`$$
A^{-1}A = I_{n} \\
$$`

> 矩阵`$A$`的**矩阵逆(matrix inversion)** 记作 `$A^{-1}$`, 其定义的矩阵满足如下条件:
> `$$A^{-1}A = I_{n}$$`

We can now solve equation `$Ax=b$` by the following steps:

`$$
Ax=b
\Rightarrow A^{-1}Ax=A^{-1}b
\Rightarrow x = A^{-1}b \\
$$`

When `$A^{-1}$` exists, several different algorithms exist for finding it in closed form. In theory, the same inverse matrix can then be used to solve the equation many times for different values of `$b$` . However, `$A^{-1}$` is primarily useful as a theoretical tool, and should not actually be used in practice for most software applications. Because `$A^{-1}$` can be represented with only limited precision on a digital computer, algorithms that make use of the value of `$b$` can usually obtain more accurate estimates of `$x$`.

## Linear Dependence and Span

In order for `$A^{-1}$` to exist, equation `$Ax=b$` must have exactly one solution for every value of `$b$`. However, it is also possible for the system of equations to have no solutions or infinitely many solutions for some values of `$b$`. It is not possible to have more than one but less than infinitely many solutions for a particular `$b$`; if both `$x$` and `$y$` are solutions then

`$$
z = \alpha x + (1-\alpha)y \\
$$`

is also a solution for any real `$\alpha$`.

To analyze how many solutions the equation has, we can think of the columns of `$A$` as specifying different directions we can travel from the **origin** (the point specified by the vector of all zeros), and determine how many ways there are of reaching `$b$`. In this view, each element of `$x$` specifies how far we should travel in each of these directions, with `$x_{i}$` specifying how far to move in the direction of column `$i$`:

`$$
Ax = \sum_{i}x_{i}A_{:,i} \\
$$`

In general, this kind of operation is called a **linear combination**. Formally, a linear combination of some set of vectors `$\{v^{(1)}, \cdots, v^{(n)}\}$` is given by multiplying each vector `$v^{(i)}$` by a corresponding scalar coefficient and adding the results:

`$$
\sum_{i}c_{i}v^{(i)} \\
$$`

The **span** of a set of vectors is the set of all points obtainable by linear combination of the original vectors.

> 一组向量的**生成子空间(span)** 是原始向量线性组合后所能抵达的点的集合.

Determining whether `$Ax=b$` has a solution thus amounts to testing whether `$b$` is in the span of the columns of `$A$`. This particular span is known as the **column space** or the **range** of `$A$`.

A set of vectors is **linearly independent** if no vector in the set is a linear combination of the other vectors. If we add a vector to a set that is a linear combination of the other vectors in the set, the new vector does not add any points to the set’s span. This means that for the column space of the matrix to encompass all of `$\mathbb{R}^{m}$`, the matrix must contain at least one set of `$m$` linearly independent columns. This condition is both necessary and sufficient for equation `$Ax=b$` to have a solution for every value of `$b$`. Note that the requirement is for a set to have exactly `$m$` linear independent columns, not at least `$m$`. No set of `$m$`-dimensional vectors can have more than m mutually linearly independent columns, but a matrix with more than `$m$` columns may have more than one such set.

> 如果一组向量中的任意一个向量都不能表示成其他向量的线性组合，那么这组向量称为**线性无关 (linearly independent)** .

In order for the matrix to have an inverse, we additionally need to ensure that equation `$Ax=b$` has at most one solution for each value of `$b$`. To do so, we need to ensure that the matrix has at most `$m$` columns. Otherwise there is more than one way of parametrizing each solution.

Together, this means that the matrix must be **square**, that is, we require that `$m=n$` and that all of the columns must be linearly independent. A square matrix with linearly dependent columns is known as **singular**.

If `$A$` is not square or is square but singular, it can still be possible to solve the equation. However, we can not use the method of matrix inversion to find the  solution.

## Norms

Sometimes we need to measure the size of a vector. In machine learning, we usually measure the size of vectors using a function called a **norm**. Formally, the `$L^{p}$` norm is given by

`$$
\lVert x \rVert_{p} = \left(\sum_{i}|x_{i}|^{p}\right)^{\frac{1}{p}} \\
$$`

for `$p \in \mathbb{R}$`, `$p>1$`.

> 我们经常使用被称为**范数 (norm)** 的函数衡量向量大小. 范数是将向量映射到非负值的函数. 直观上来说, 向量 `$x$` 的范数衡量从原点到点 `$x$` 的距离.

Norms, including the `$L^{p}$` norm, are functions mapping vectors to non-negative values. On an intuitive level, the norm of a vector `$x$` measures the distance from the origin to the point `$x$`. More rigorously, a norm is any function `$f$` that satisfies the following properties:

* `$f(x) = 0 \Rightarrow x = 0$`
* `$f(x+y) \leq f(x) + f(y)$` (the triangle inequality)
* `$\forall \alpha \in \mathbb{R}, f(\alpha x) = |\alpha|f(x)$`

> **范数 (norm)** 满足以下性质：
>* `$f(x) = 0 \Rightarrow x = 0$` <br>
>* `$f(x+y) \leq f(x) + f(y)$` (满足三角不等式, 或称次可加性)<br>
>* `$\forall \alpha \in \mathbb{R}, f(\alpha x) = |\alpha|f(x)$` (具有绝对一次齐次性)<br>

The `$L^{2}$` norm, with `$p=2$`, is known as the **Euclidean norm**. It is simply the Euclidean distance from the origin to the point identified by `$x$`. The `$L^{2}$` norm is used so frequently in machine learning that it is often denoted simply as `$\lVert x \rVert$`, with the subscript 2 omitted. It is also common to measure the size of a vector using the squared `$L^{2}$` norm, which can be calculated simply as `$x^{T}x$`.

`$$
\lVert x \rVert_{2} = \sqrt{\sum_{i}|x_{i}|^{2}} \\
$$`

> 当 `$p=2$` 时，`$L^{2}$` 范数被称为**欧几里得范数(Euclidean norm)** .它表示从原点出发到向量 `$x$` 确定的点的欧几里得距离.

The `$L^{1}$` norm is commonly used in machine learning when the difference between zero and nonzero elements is very important. Every time an element of `$x$` moves away from 0 by `$\varepsilon$`, the `$L^{1}$` norm increases by `$\varepsilon$`. The `$L^{1}$` norm may be simplified to

`$$
\lVert x \rVert_{1} = \sum_{i}|x_{i}| \\
$$`

One other norm that commonly arises in machine learning is the `$L^{\infty}$` norm, also known as the **max norm**. This norm simplifies to the absolute value of the element with the largest magnitude in the vector,

`$$
\lVert x \rVert_{\infty} = max_{i}|x_{i}| \\
$$`

Sometimes we may also wish to measure the size of a matrix. In the context of deep learning, the most common way to do this is with the otherwise obscure
**Frobenius norm**:

`$$
\lVert x \rVert_{F} = \sqrt{\sum_{i,j}A_{i,j}^{2}} \\
$$`

The dot product of two vectors can be rewritten in terms of norms. Specifically,

`$$
x^{T}y = \lVert x \rVert_{2}\lVert y \rVert_{2} \cos{\theta} \\
$$`

where `$\theta$` is the angle between `$x$` and `$y$`.

## Special Kinds of Matrices and Vectors

Some special kinds of matrices and vectors are particularly useful.

**Diagonal matrices** consist mostly of zeros and have non-zero entries only along the main diagonal. Formally, a matrix `$D$` is diagonal if and only if `$D_{i,j}=0$` for all `$i \neq j$`. We have already seen one example of a diagonal matrix: the identity matrix, where all of the diagonal entries are 1. We write `$diag(v)$` to denote a square diagonal matrix whose diagonal entries are given by the entries of the vector `$v$`, e.g.,

`$$
v = \begin{bmatrix}
a_{1} \\
a_{2} \\
a_{3} \\
\end{bmatrix} \Rightarrow diag(v) = \begin{bmatrix}
a_{1} & 0 & 0 \\
0 & a_{2} & 0 \\
0 & 0 & a_{3} \\
\end{bmatrix}
$$`

Diagonal matrices are of interest in part because multiplying by a diagonal matrix is very **computationally efficient**. To compute `$diag(v)x$`, we only need to scale each element `$x_{i}$` by `$v_{i}$`. In other words, `$diag(v)x = v \odot x$`. Inverting a square diagonal matrix is also efficient. The inverse exists only if every diagonal entry is nonzero, `$diag(v)^{-1}= diag([\frac{1}{v_{1}}, \cdots, \frac{1}{v_{n}}]^{T})$`.

> **对角矩阵(diagonal matrix)** 只在主对角线上含有非零元素, 其他位置都是零.

A **symmetric matrix** is any matrix that is equal to its own transpose:

`$$
A = A^{T} \\
$$`

For example,

`$$
A = A^{T} = \begin{bmatrix}
1 & 2 \\
2 & 3 \\
\end{bmatrix} \ \ \ \ \ \ \ \ B = B^{T} = \begin{bmatrix}
5 & 6 & 7 \\
6 & 3 & 2 \\
7 & 2 & 1 \\
\end{bmatrix}
$$`

Symmetric matrices often arise when the entries are generated by some function of two arguments that does not depend on the order of the arguments. For example, if `$A$` is a matrix of distance measurements, with `$A_{i,j}$` giving the distance from point `$i$` to point `$j$`, then `$A_{i,j} = A_{j,i}$` because distance functions are symmetric.

> **对称矩阵(symmetric matrix)** 是转置和自己相等的矩阵.

A **unit vector** is a vector with unit norm:

`$$
\lVert x \rVert_{2} = 1 \\
$$`

A vector `$x$` and a vector `$y$` are orthogonal to each other if `$x^{T}y=0$`. If both vectors have nonzero norm, this means that they are at a 90 degree angle to each other. In `$\mathbb{R}^{n}$`, at most `$n$` vectors may be mutually orthogonal with nonzero norm. If the vectors are not only orthogonal but also have unit norm, we call them **orthonormal**.

> **单位向量(unit vector)** 是具有**单位范数(unit norm)** 的向量. 如果 `$x^{T}y=0$`，那么向量 `$x$` 和向量 `$y$` 互相**正交(orthogonal)** . 如果两个向量都有非零范数, 那么这两个向量之间的夹角是90度. 如果这些向量不仅互相正交，并且范数都为 1，那么我们称它们是**标准正交(orthonormal)** .

An **orthogonal matrix** is a square matrix whose rows are mutually orthonormal and whose columns are mutually orthonormal:

`$$
A^{T}A = AA^{T} = I \\
$$`

This implies that

`$$
A^{-1} = A^{T} \\
$$`

so orthogonal matrices are of interest because their inverse is very cheap to compute. Pay careful attention to the definition of orthogonal matrices. Counterintuitively, their rows are not merely orthogonal but fully orthonormal. There is no special term for a matrix whose rows or columns are orthogonal but not orthonormal.

## Eigendecomposition

Much as we can discover something about the true nature of an integer by decomposing it into prime factors, we can also decompose matrices in ways that show us information about their functional properties that is not obvious from the representation of the matrix as an array of elements.

One of the most widely used kinds of matrix decomposition is called **eigendecomposi-tion**, in which we decompose a matrix into a set of eigenvectors and eigenvalues.

An **eigenvector** of a square matrix `$A$` is a non-zero vector `$v$` such that multiplication by `$A$` alters only the scale of `$v$`:

`$$
Av = \lambda v \\
$$`

The scalar `$\lambda$` is known as the **eigenvalue** corresponding to this eigenvector. If `$v$` is an eigenvector of `$A$`, then so is any rescaled vector `$sv$` for `$s \in \mathbb{R}$`, `$s \neq 0$`. Moreover, `$sv$` still has the same eigenvalue. For this reason, we usually only look for unit eigenvectors.

> **特征分解(eigendecomposition)** 是使用最广的矩阵分解之一, 即我们将矩阵分解成一组特征向量和特征值. 方阵 `$A$` 的**特征向量(eigenvector)** 是指与 `$A$` 相乘后相当于对该向量进行缩放的非零向量 `$v$`:
> `$$Av = \lambda v \\$$`
> 标量 `$\lambda$` 被称为这个特征向量对应的**特征值(eigenvalue)** .

Suppose that a matrix `$A$` has `$n$` linearly independent eigenvectors, `$\{v^{(1)}, \cdots, v^{(n)}\}$`, with corresponding eigenvalues `$\{\lambda_{1}, \cdots, \lambda_{n}\}$`. We may concatenate all of the eigenvectors to form a matrix `$V$` with one eigenvector per column: `$V=[v^{(1)}, \cdots, v^{(n)}]$`. Likewise, we can concatenate the eigenvalues to form a vector `$\lambda = [\lambda_{1}, \cdots, \lambda_{n}]^{T}$`. The **eigendecomposi-tion** of `$A$` is then given by

`$$
A = V diag(\lambda)V^{-1} \\
$$`

> `$A$`的**特征分解(eigendecomposition)** 可以记作:
> `$$A = V diag(\lambda)V^{-1} $$`

We have seen that constructing matrices with specific eigenvalues and eigenvectors allows us to stretch space in desired directions. However, we often want to decompose matrices into their eigenvalues and eigenvectors. Doing so can help us to analyze certain properties of the matrix, much as **decomposing** an integer into its prime factors can help us understand the behavior of that integer. Not every matrix can be decomposed into eigenvalues and eigenvectors. In some cases, the decomposition exists, but may involve complex rather than real numbers.

## Singular Value Decomposition

We already known how to decompose a matrix into eigenvectors and eigenvalues. The **singular value decomposition** (SVD) provides another way to factorize a matrix, into **singular vectors** and **singular values**. The SVD allows us to discover some of the same kind of information as the eigendecomposition. However, the SVD is more generally applicable. Every real matrix has a singular value decomposition, but the same is not true of the eigenvalue decomposition. For example, if a matrix is not square, the eigendecom-position is not defined, and we must use a singular value decomposition instead.

Recall that the eigendecomposition involves analyzing a matrix `$A$` to discover a matrix `$V$` of eigenvectors and a vector of eigenvalues `$\lambda$` such that we can rewrite `$A$` as

`$$
A = V diag(\lambda)V^{-1} \\
$$`

The singular value decomposition is similar, except this time we will write `$A$` as a product of three matrices:

`$$
A = UDV^{T} \\
$$`

Suppose that `$A$` is an `$m \times n$` matrix. Then `$U$` is defined to be an `$m \times m$` matrix, `$D$` to be an `$m \times n$` matrix, and `$V$` to be an `$n \times n$` matrix.

> **奇异值分解(singular value decomposition, SVD)**, 将矩阵分解为**奇异向量(singular vector)** 和**奇异值(singular value)** :
> `$$A = UDV^{T} \\$$`
>  如果 `$A$` 是一个 `$m \times n$` 的矩阵. 那么 `$U$` 是一个 `$m \times m$` 的矩阵, `$D$` 是一个 `$m \times n$` 的矩阵, `$V$` 是一个 `$n \times n$` 的矩阵.

Each of these matrices is defined to have a special structure. The matrices `$U$` and `$V$` are both defined to be orthogonal matrices. The matrix `$D$` is defined to be a diagonal matrix. Note that `$D$` is not necessarily square.

The elements along the diagonal of `$D$` are known as the **singular values** of the matrix `$A$`. The columns of `$U$` are known as the **left-singular vectors**. The columns of `$V$` are known as as the **right-singular vectors**.

## The Trace Operator

The trace operator gives the sum of all of the diagonal entries of a matrix:

`$$
Tr(A) = \sum_{i}A_{i,i} \\
$$`

> **迹运算(trace operator)** 返回的是矩阵对角元素的和.

For example, let `$A$` be a matrix, with

`$$
A = \begin{bmatrix}
a_{1,1} & a_{1,2}  \\
a_{2,1} & a_{2,2}  \\
\end{bmatrix}
$$`

Then,

`$$
Tr(A) = \sum_{i=1}^{2}a_{i,i} = a_{1,1} + a_{2,2}
$$`

The trace operator is useful for a variety of reasons. Some operations that are difficult to specify without resorting to summation notation can be specified using  matrix products and the trace operator. For example, the trace operator provides an alternative way of writing the Frobenius norm of a matrix:

`$$
\lVert x \rVert_{F} = \sqrt{Tr(AA^{T})} \\
$$`

Writing an expression in terms of the trace operator opens up opportunities to manipulate the expression using many useful identities. For example, the trace operator is invariant to the transpose operator:

`$$
Tr(A) = Tr(A^{T}) \\
$$`

The trace of a square matrix composed of many factors is also invariant to moving the last factor into the first position, if the shapes of the corresponding matrices allow the resulting product to be defined:

`$$
Tr(ABC) = Tr(CAB) = Tr(BCA) \\
$$`

This invariance to cyclic permutation holds even if the resulting product has a different shape. For example, for `$A \in \mathbb{R}^{m \times n}$` and `$B \in \mathbb{R}^{n \times m}$`, we have

`$$
Tr(AB) = Tr(BA) \\
$$`

even though  `$AB \in \mathbb{R}^{m \times m}$` and `$BA \in \mathbb{R}^{n \times n}$`

Another useful fact to keep in mind is that a scalar is its own trace: `$a = Tr(a)$`.

## The Determinant

The determinant of a square matrix, denoted `$det(A)$`, is a function mapping matrices to real scalars. The determinant is equal to the product of all the eigenvalues of the matrix. The absolute value of the determinant can be thought of as a measure of how much multiplication by the matrix expands or contracts space. If the determinant is 0, then space is contracted completely along at least one dimension, causing it to lose all of its volume. If the determinant is 1, then the transformation preserves volume.

> **行列式(determinant)** 是一个将方阵映射到实数的函数. 行列式等于矩阵特征值的乘积.

For example, for a `$2 \times 2$` matrix:

`$$
A = \begin{bmatrix}
a & b  \\
c & d  \\
\end{bmatrix}
$$`

The determinant is:

`$$
det(A) = |A| = ad - bc
$$`

## Jacobian and Hessian Matrices

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

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/linear_algebra.html.
