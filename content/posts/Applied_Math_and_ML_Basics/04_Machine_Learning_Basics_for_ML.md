---
title: "[ML Basics] Machine Learning Basics"
date: "2021-06-30"
tags: ["Machine Learning"]
categories: ["Machine Learning Basics", "Data Science"]
weight: 3
---

## Learning Algorithms

A machine learning algorithm is an algorithm that is able to learn from data. But what do we mean by learning? Mitchell (1997) provides the definition "A computer program is said to learn from experience `$E$` with respect to some class of tasks `$T$` and performance measure `$P$`, if its performance at tasks in `$T$`, as measured by `$P$`, improves with experience `$E$`."

### The Task, `$T$`

Machine learning allows us to tackle tasks that are too difficult to solve with fixed programs written and designed by human beings. From a scientific and philosophical point of view, machine learning is interesting because developing our understanding of machine learning entails developing our understanding of the principles that underlie intelligence.

In this relatively formal definition of the word "task," the process of learning itself is not the task. Learning is our means of attaining the ability to perform the task.

Machine learning tasks are usually described in terms of how the machine learning system should process an **example**. An example is a collection of **features** that have been quantitatively measured from some object or event that we want the machine learning system to process. We typically represent an example as a vector `$x \in \mathbb{R}$` where each entry `$x_{i}$` of the vector is another feature. For example, the features of an image are usually the values of the pixels in the image.

> 通常机器学习任务定义为机器学习系统应该如何处理 **样本(example)**. 样本是指我们从某些希望机器学习系统处理的对象或事件中收集到的已经量化的 **特征 (feature)** 的集合.

Many kinds of tasks can be solved with machine learning. Some of the most common machine learning tasks include the following:

*  **Classification**: In this type of task, the computer program is asked to specify which of `$k$` categories some input belongs to. To solve this task, the learning algorithm is usually asked to produce a function `$f: \mathbb{R}^{n} \to \{1, \cdots,k \}$`. When `$y=f(x)$`, the model assigns an input described by vector `$x$` to a category identified by numeric code `$y$`. There are other variants of the classification task, for example, where `$f$` outputs a probability distribution over classes. An example of a classification task is **object recognition**, where the input is an image (usually described as a set of pixel brightness values), and the output is a numeric code identifying the object in the image. Object recognition is the same basic technology that allows computers to recognize faces (Taigman et al., 2014), which can be used to automatically tag people in photo collections and allow computers to interact more naturally with their users.

> **分类**: 在这类任务中, 计算机程序需要指定某些输入属于 `$k$` 类中的哪一类. 为了完成这个任务, 学习算法通常会返回一个函数 `$f: \mathbb{R}^{n} \to \{1, \cdots,k \}$`.当 `$y=f(x)$` 时, 模型将向量 `$x$` 所代表的输入分类到数字码 `$y$` 所代表的类别.还有一些其他的分类问题, 例如, `$f$` 输出的是不同类别的概率分布. 代表示例有： 人脸识别.

* **Classification with missing inputs**: Classification becomes more challenging if the computer program is not guaranteed that every measurement in its input vector will always be provided. In order to solve the classification task, the learning algorithm only has to define a *single* function mapping from a vector input to a categorical output. When some of the inputs may be missing, rather than providing a single classification function, the learning algorithm must learn a *set* of functions. Each function corresponds to classifying `$x$` with a different subset of its inputs missing. This kind of situation arises frequently in medical diagnosis, because many kinds of medical tests are expensive or invasive. One way to efficiently define such a large set of functions is to learn a probability distribution over all of the relevant variables, then solve the classification task by marginalizing out the missing variables. With `$n$` input variables, we can now obtain all `$2^{n}$` different classification functions needed for each possible set of missing inputs, but we only need to learn a single function describing the joint probability distribution.

> **输入缺失分类**: 当输入向量的每个度量不被保证的时候, 分类问题将会变得更有挑战性. 为了解决分类任务, 学习算法只需要定义一个从输入向量映射到输出类别的函数. 当一些输入可能丢失时, 学习算法必须学习一组函数, 而不是单个分类函数. 每个函数对应着分类具有不同缺失输入子集的 `$x$`. 这种情况在 医疗诊断中经常出现, 因为很多类型的医学测试是昂贵的, 对身体有害的.

* **Regression**: In this type of task, the computer program is asked to predict a numerical value given some input. To solve this task, the learning algorithm is asked to output a function `$f:\mathbb{R}^{n}\to\mathbb{R}$`. This type of task is similar to classification, except that the format of output is different. An example of a regression task is the prediction of the expected claim amount that an insured person will make (used to set insurance premiums), or the prediction of future prices of securities. These kinds of predictions are also used for algorithmic trading.

> **回归**: 在这类任务中, 计算机程序需要对给定输入预测数值.为了解决这个任务, 学习算法需要输出函数 `$f:\mathbb{R}^{n}\to\mathbb{R}$`.除了返回结果的形式不一样外, 这类问题和分类问题是很像的. 代表示例有：预测证券未来的价格.

* **Transcription**: In this type of task, the machine learning system is asked to observe a relatively unstructured representation of some kind of data and transcribe it into discrete, textual form. For example, in optical character recognition, the computer program is shown a photograph containing an image of text and is asked to return this text in the form of a sequence of characters (e.g., in ASCII or Unicode format). Another example is speech recognition, where the computer program is provided an audio waveform and emits a sequence of characters or word ID codes describing the words that were spoken in the audio recording. Deep learning is a crucial component of modern speech recognition systems used at major companies.

> **转录**: 这类任务中, 机器学习系统观测一些相对非结构化表示的数据, 并转录信息为离散的文本形式.代表示例有：光学字符识别要求计算机程序根据文本图片返回文字序列、语音识别.

* **Machine translation**: In a machine translation task, the input already consists of a sequence of symbols in some language, and the computer program must convert this into a sequence of symbols in another language. This is commonly applied to natural languages, such as translating from English to French.

> **机器翻译**: 在机器翻译任务中, 输入是一种语言的符号序列, 计算机程序必须将其转化成另一种语言的符号序列. 代表示例有：英语翻译成法语.

* **Structured output**: Structured output tasks involve any task where the output is a vector (or other data structure containing multiple values) with important relationships between the different elements. This is a broad category, and subsumes the transcription and translation tasks described above, but also many other tasks. One example is parsing—mapping a natural language sentence into a tree that describes its grammatical structure and tagging nodes of the trees as being verbs, nouns, or adverbs, and so on. Another example is pixel-wise segmentation of images, where the computer program assigns every pixel in an image to a specific category.

> **结构化输出**: 结构化输出任务的输出是向量或者其他包含多个值的数据结构, 并且构成输出的这些不同元素间具有重要关系.这是一个很大的范畴, 包括上述转录任务和翻译任务在内的很多其他任务. 代表示例有：语法分析、图像的像素级分割, 将每一个像素分配到特定类别.

* **Anomaly detection**: In this type of task, the computer program sifts through a set of events or objects, and flags some of them as being unusual or atypical. An example of an anomaly detection task is credit card fraud detection. By modeling your purchasing habits, a credit card company can detect misuse of your cards. If a thief steals your credit card or credit card information, the thief’s purchases will often come from a different probability distribution over purchase types than your own. The credit card company can prevent fraud by placing a hold on an account as soon as that card has been used for an uncharacteristic purchase.

> **异常检测**: 在这类任务中, 计算机程序在一组事件或对象中筛选, 并标记不正常或非典型的个体. 异常检测任务的一个示例是信用卡欺诈检测.

* **Synthesis and sampling**: In this type of task, the machine learning algorithm is asked to generate new examples that are similar to those in the training data. Synthesis and sampling via machine learning can be useful for media applications where it can be expensive or boring for an artist to generate large volumes of content by hand. For example, video games can automatically generate textures for large objects or landscapes, rather than requiring an artist to manually label each pixel.

> **合成和采样**: 在这类任务中, 机器学习程序生成一些和训练数据相似的新样本. 通过机器学习, 合成和采样可能在媒体应用中非常有用, 可以避免艺术家大量昂贵或者乏味费时的手动工作. 例如, 视频游戏可以自动生成大型物体或风景的纹理, 而不是让艺术家手动标记每个像素.

* **Imputation of missing values**: In this type of task, the machine learning algorithm is given a new example `$x \in \mathbb{R}^{n}$`, but with some entries `$x_{i}$` of `$x$` missing. The algorithm must provide a prediction of the values of the missing entries.

> **缺失值填补**: 在这类任务中, 机器学习算法给定一个新样本 `$x \in \mathbb{R}^{n}$`, `$x$` 中某些元素 `$x_{i}$` 缺失.算法必须填补这些缺失值.

* **Denoising**: In this type of task, the machine learning algorithm is given in input a *corrupted example* `$\tilde{x} \in \mathbb{R}^{n}$` obtained by an unknown corruption process from a *clean example* `$x\in \mathbb{R}^{n}$`. The learner must predict the clean example `$x$` from its corrupted version `$\tilde{x}$`, or more generally predict the conditional probability distribution `$p(x \ | \ \tilde{x})$`.

> **去噪**: 在这类任务中, 机器学习算法的输入是, 干净样本 `$x\in \mathbb{R}^{n}$` 经过未知损坏过程后得到的损坏样本 `$\tilde{x} \in \mathbb{R}^{n}$`.算法根据损坏后的样本 `$\tilde{x}$` 预测干净的样本 `$x$`, 或者更一般地预测条件概率分布 `$p(x \ | \ \tilde{x})$`.

Of course, many other tasks and types of tasks are possible. The types of tasks we list here are intended only to provide examples of what machine learning can do, not to define a rigid taxonomy of tasks.

### The Performance Measure, `$P$`

In order to evaluate the abilities of a machine learning algorithm, we must design a quantitative measure of its performance. Usually this performance measure `$P$` is specific to the task `$T$` being carried out by the system.

For tasks such as classification, classification with missing inputs, and transcription, we often measure the **accuracy** of the model. Accuracy is just the proportion of examples for which the model produces the correct output. We can also obtain equivalent information by measuring the **error rate**, the proportion of examples for which the model produces an incorrect output. We often refer to the error rate as the expected 0-1 loss. The 0-1 loss on a particular example is 0 if it is correctly classified and 1 if it is not. For tasks such as density estimation, it does not make sense to measure accuracy, error rate, or any other kind of 0-1 loss. Instead, we must use a different performance metric that gives the model a continuous-valued score for each example. The most common approach is to report the average log-probability the model assigns to some examples.

> 对于诸如分类、缺失输入分类和转录任务, 我们通常度量模型的 **准确率(accuracy)**.准确率是指该模型输出正确结果的样本比率. 我们也可以通过 **错误率(error rate)** 得到相同的信息. 错误率是指该模型输出错误结果的样本比率.

Usually we are interested in how well the machine learning algorithm performs on data that it has not seen before, since this determines how well it will work when deployed in the real world. We therefore evaluate these performance measures using a **test set** of data that is separate from the data used for training the machine learning system.

The choice of performance measure may seem straightforward and objective, but it is often difficult to choose a performance measure that corresponds well to the desired behavior of the system.

### The Experience, `$E$`

Machine learning algorithms can be broadly categorized as **unsupervised** or **supervised** by what kind of experience they are allowed to have during the learning process.

Most of the learning algorithms can be understood as being allowed to experience an entire **dataset**. A dataset is a collection of many examples. Sometimes we will also call examples **data points**.

**Unsupervised learning algorithms** experience a dataset containing many features, then learn useful properties of the structure of this dataset. In the context of deep learning, we usually want to learn the entire probability distribution that generated a dataset, whether explicitly as in density estimation or implicitly for tasks like synthesis or denoising. Some other unsupervised learning algorithms perform other roles, like clustering, which consists of dividing the dataset into clusters of similar examples.

> **无监督学习算法(unsupervised learning algorithm)** 中, 数据没有标签.无监督学习使我们能够在几乎不知道结果应该是什么样子的情况下解决问题. 我们可以从数据中推导出结构, 而我们不一定知道变量的影响. 我们可以通过基于数据中变量之间的关系对数据进行聚类来推导出这种结构. 对于无监督学习, 没有基于预测结果的反馈.

**Supervised learning algorithms** experience a dataset containing features, but each example is also associated with a **label** or **target**.

> **监督学习算法(supervised learning algorithm)** 中, 我们得到了一个数据集, 并且已经知道我们的正确输出应该是什么样子, 并且知道输入和输出之间存在关系.
> 监督学习问题分为“回归”和“分类”问题. 在回归问题中, 我们试图预测连续输出中的结果, 这意味着我们试图将输入变量映射到某个连续函数.在分类问题中, 我们试图在离散输出中预测结果. 换句话说, 我们试图输入变量映射到离散类别中.

Unsupervised learning and supervised learning are not formally defined terms. The lines between them are often blurred. Many machine learning technologies can be used to perform both tasks.

Though unsupervised learning and supervised learning are not completely formal or distinct concepts, they do help to roughly categorize some of the things we do with machine learning algorithms. Traditionally, people refer to regression, classification and structured output problems as supervised learning. Density estimation in support of other tasks is usually considered unsupervised learning.

Other variants of the learning paradigm are possible. For example, in **semi-supervised learning**, some examples include a supervision target but others do not. In multi-instance learning, an entire collection of examples is labeled as containing or not containing an example of a class, but the individual members of the collection are not labeled.

> **半监督学习(semi-supervised learning)** 中, 一些样本有监督目标, 但其他样本没有.

Some machine learning algorithms do not just experience a fixed dataset. For example, **reinforcement learning algorithms** interact with an environment, so there is a feedback loop between the learning system and its experiences.

> **强化算法(reinforcement learning algorithms)** 通过反复试验来实现明确的目标. 它尝试了许多不同的事情, 并根据其行为是帮助还是阻碍其实现目标而受到奖励或惩罚. 这就像在教狗新把戏时给予和扣留零食一样. 强化学习是谷歌 AlphaGo 的基础.

Most machine learning algorithms simply experience a dataset. A dataset can be described in many ways. In all cases, a dataset is a collection of examples, which are in turn collections of features.

## Capacity, Overfitting and Underfitting

The central challenge in machine learning is that we must perform well on new, previously unseen inputs—not just those on which our model was trained. The ability to perform well on previously unobserved inputs is called **generalization**.

> 机器学习的主要挑战是我们的算法必须能够在先前未观测的新输入上表现良好, 而不只是在训练集上表现良好.在先前未观测到的输入上表现良好的能力被称为 **泛化(generalization)** .

Typically, when training a machine learning model, we have access to a training set, we can compute some error measure on the training set called the **training error**, and we reduce this training error. So far, what we have described is simply an optimization problem. What separates machine learning from optimization is that we want the **generalization error**, also called the **test error**, to be low as well. The generalization error is defined as the expected value of the error on a new input. Here the expectation is taken across different possible inputs, drawn from the distribution of inputs we expect the system to encounter in practice.

> 通常情况下, 当我们训练机器学习模型时, 我们可以使用某个训练集, 在训练集上计算一些被称为 **训练误差(training error)** 的度量误差, 目标是降低训练误差.机器学习和优化不同的地方在于, 我们也希望 **泛化误差(generalization error)** (也被称为 **测试误差(test error)** )很低.

We typically estimate the generalization error of a machine learning model by measuring its performance on a **test set** of examples that were collected separately from the training set.

The train and test data are generated by a probability distribution over datasets called the **data generating process**. We typically make a set of assumptions known collectively as the **i.i.d. assumptions**. These assumptions are that the examples in each dataset are **independent** from each other, and that the train set and test set are **identically distributed**, drawn from the same probability distribution as each other. This assumption allows us to describe the data generating process with a probability distribution over a single example. The same distribution is then used to generate every train example and every test example. We call that shared underlying distribution the **data generating distribution**, denoted `$p_{data}$`. This probabilistic framework and the i.i.d. assumptions allow us to mathematically study the relationship between training error and test error.

One immediate connection we can observe between the training and test error is that the expected training error of a randomly selected model is equal to the expected test error of that model. Suppose we have a probability distribution `$p(x,y)$` and we sample from it repeatedly to generate the train set and the test set. For some fixed value `$w$`, the expected training set error is exactly the same as the expected test set error, because both expectations are formed using the same dataset sampling process. The only difference between the two conditions is the name we assign to the dataset we sample.

Of course, when we use a machine learning algorithm, we do not fix the parameters ahead of time, then sample both datasets. We sample the training set, then use it to choose the parameters to reduce training set error, then sample the test set. Under this process, the expected test error is greater than or equal to the expected value of training error. The factors determining how well a machine learning algorithm will perform are its ability to:

1. Make the training error small.
2. Make the gap between training and test error small.

These two factors correspond to the two central challenges in machine learning: **underfitting** and **overfitting**. Underfitting occurs when the model is not able to obtain a sufficiently low error value on the training set. Overfitting occurs when the gap between the training error and test error is too large.

> **欠拟合(underfitting)** 是指模型不能在训练集上获得足够低的误差. 而 **过拟合 (overfitting)** 是指训练误差和和测试误差之间的差距太大.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_04_Underfitting_Overfitting.PNG" width=500px/>
</div>
<br>

We can control whether a model is more likely to overfit or underfit by altering its **capacity**. Informally, a model’s capacity is its ability to fit a wide variety of functions. Models with low capacity may struggle to fit the training set. Models with high capacity can overfit by memorizing properties of the training set that do not serve them well on the test set.

One way to control the capacity of a learning algorithm is by choosing its **hypothesis space**, the set of functions that the learning algorithm is allowed to select as being the solution. For example, the linear regression algorithm has the set of all linear functions of its input as its hypothesis space. We can generalize linear regression to include polynomials, rather than just linear functions, in its hypothesis space. Doing so increases the model’s capacity.

We must remember that while simpler functions are more likely to generalize (to have a small gap between training and test error) we must still choose a sufficiently complex hypothesis to achieve low training error. Typically, training error decreases until it asymptotes to the minimum possible error value as model capacity increases (assuming the error measure has a minimum value). Typically, generalization error has a U-shaped curve as a function of model capacity.

<div align="center">
  <img src="/img_ML_Basics/ML_Basics_04_Relationship_between_Capacity_Error.PNG" width=600px/>
</div>
<br>

### The No Free Lunch Theorem

Learning theory claims that a machine learning algorithm can generalize well from a finite training set of examples. This seems to contradict some basic principles of logic. Inductive reasoning, or inferring general rules from a limited set of examples, is not logically valid. To logically infer a rule describing every member of a set, one must have information about every member of that set.

In part, machine learning avoids this problem by offering only probabilistic rules, rather than the entirely certain rules used in purely logical reasoning. Machine learning promises to find rules that are probably correct about most members of the set they concern.

Unfortunately, even this does not resolve the entire problem. The **no free lunch theorem** for machine learning (Wolpert, 1996) states that, averaged over all possible data generating distributions, every classification algorithm has the same error rate when classifying previously unobserved points. In other words, in some sense, no machine learning algorithm is universally any better than any other. The most sophisticated algorithm we can conceive of has the same average performance (over all possible tasks) as merely predicting that every point belongs to the same class.

> 机器学习的 没有 **免费午餐定理(no free lunch theorem)** 表明 (Wolpert, 1996), 在所有可能的数据生成分布上平均之后, 每一个分类算法在未事先观测的点上都有相同的错误率. 换言之, 在某种意义上, 没有一个机器学习算法总是比其他的要好.

Fortunately, these results hold only when we average over all possible data generating distributions. If we make assumptions about the kinds of probability distributions we encounter in real-world applications, then we can design learning algorithms that perform well on these distributions.

This means that the goal of machine learning research is not to seek a universal learning algorithm or the absolute best learning algorithm. Instead, our goal is to understand what kinds of distributions are relevant to the "real world" that an AI agent experiences, and what kinds of machine learning algorithms perform well on data drawn from the kinds of data generating distributions we care about.

### Regularization

The no free lunch theorem implies that we must design our machine learning algorithms to perform well on a specific task. We do so by building a set of preferences into the learning algorithm. When these preferences are aligned with the learning problems we ask the algorithm to solve, it performs better.

We can regularize a model that learns a function f(x; θ) by adding a penalty called a **regularizer** to the cost function.

Expressing preferences for one function over another is a more general way of controlling a model’s capacity than including or excluding members from the hypothesis space. We can think of excluding a function from a hypothesis space as expressing an infinitely strong preference against that function.

In weight decay example, we expressed our preference for linear functions defined with smaller weights explicitly, via an extra term in the criterion we minimize. There are many other ways of expressing preferences for different solutions, both implicitly and explicitly. Together, these different approaches are known as **regularization**. *Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error*. Regularization is one of the central concerns of the field of machine learning, rivaled in its importance only by optimization.

> **正则化(regularization)** 是指我们修改学习算法，使其降低泛化误差而非训练误差。正则化是机器学习领域的中心问题之一，只有优化能够与其重要性相媲。

The no free lunch theorem has made it clear that there is no best machine learning algorithm, and, in particular, no best form of regularization. Instead we must choose a form of regularization that is well-suited to the particular task we want to solve.

## Hyperparameters and Validation Sets

Most machine learning algorithms have several settings that we can use to control the behavior of the learning algorithm. These settings are called hyperparame- ters. The values of hyperparameters are not adapted by the learning algorithm itself (though we can design a nested learning procedure where one learning algorithm learns the best hyperparameters for another learning algorithm).

Sometimes a setting is chosen to be a hyperparameter that the learning algorithm does not learn because it is difficult to optimize. More frequently, the setting must be a hyperparameter because it is not appropriate to learn that hyperparameter on the training set. This applies to all hyperparameters that control model capacity. If learned on the training set, such hyperparameters would always choose the maximum possible model capacity, resulting in overfitting. For example, we can always fit the training set better with a higher degree polynomial and a weight decay setting of `$\lambda = 0$` than we could with a lower degree polynomial and a positive weight decay setting.

To solve this problem, we need a **validation set** of examples that the training algorithm does not observe.

Earlier we discussed how a held-out test set, composed of examples coming from the same distribution as the training set, can be used to estimate the generalization error of a learner, after the learning process has completed. It is important that the test examples are not used in any way to make choices about the model, including its hyperparameters. For this reason, no example from the test set can be used in the validation set. Therefore, we always construct the validation set from the training data. Specifically, we split the training data into two disjoint subsets. One of these subsets is used to learn the parameters. The other subset is our validation set, used to estimate the generalization error during or after training, allowing for the hyperparameters to be updated accordingly. The subset of data used to learn the parameters is still typically called the training set, even though this may be confused with the larger pool of data used for the entire training process. The subset of data used to guide the selection of hyperparameters is called the validation set. Typically, one uses about 80% of the training data for training and 20% for validation. Since the validation set is used to "train" the hyperparameters, the validation set error will underestimate the generalization error, though typically by a smaller amount than the training error. After all hyperparameter optimization is complete, the generalization error may be estimated using the test set.

### Cross-Validation

Dividing the dataset into a fixed training set and a fixed test set can be problematic if it results in the test set being small. A small test set implies statistical uncertainty around the estimated average test error, making it difficult to claim that algorithm `$A$` works better than algorithm `$B$` on the given task.

When the dataset has hundreds of thousands of examples or more, this is not a serious issue. When the dataset is too small, are alternative procedures enable one to use all of the examples in the estimation of the mean test error, at the price of increased computational cost. These procedures are based on the idea of repeating the training and testing computation on different randomly chosen subsets or splits of the original dataset. The most common of these is the `$k$`-fold cross-validation procedure, in which a partition of the dataset is formed by splitting it into `$k$` non-overlapping subsets. The test error may then be estimated by taking the average test error across `$k$` trials. On trial `$i$`, the `$i$`-th subset of the data is used as the test set and the rest of the data is used as the training set.

## Estimators, Bias and Variance

The field of statistics gives us many tools that can be used to achieve the machine learning goal of solving a task not only on the training set but also to generalize. Foundational concepts such as parameter estimation, bias and variance are useful to formally characterize notions of generalization, underfitting and overfitting.

### Point Estimation

Point estimation is the attempt to provide the single “best” prediction of some quantity of interest. In general the quantity of interest can be a single parameter or a vector of parameters in some parametric model.

In order to distinguish estimates of parameters from their true value, our convention will be to denote a point estimate of a parameter `$\theta$` by `$\hat{\theta}$`.

Let `$\{x^{(1)},\cdots,x^{(m)}\}$` be a set of `$m$` independent and identically distributed (i.i.d.) data points. A **point estimator** or **statistic** is any function of the data:

`$$
\hat{\theta}_{m}=g(x^{(1)},\cdots,x^{(m)}) \\
$$`

The definition does not require that `$g$` return a value that is close to the true `$\theta$` or even that the range of `$g$` is the same as the set of allowable values of `$\theta$`. This definition of a point estimator is very general and allows the designer of an estimator great flexibility. While almost any function thus qualifies as an estimator, a good estimator is a function whose output is close to the true underlying `$\theta$` that generated the training data.

For now, we take the frequentist perspective on statistics. That is, we assume that the true parameter value `$\theta$` is fixed but unknown, while the point estimate `$\hat{\theta}$` is a function of the data. Since the data is drawn from a random process, any function of the data is random. Therefore `$\hat{\theta}$` is a random variable.

Point estimation can also refer to the estimation of the relationship between input and target variables. We refer to these types of point estimates as function estimators.

**Function Estimation**: As we mentioned above, sometimes we are interested in performing function estimation (or function approximation). Here we are trying to
predict a variable `$y$` given an input vector `$x$`. We assume that there is a function `$f(x)$` that describes the approximate relationship between `$y$` and `$x$`. For example, we may assume that `$y=f(x)+\epsilon$`, where `$\epsilon$` stands for the part of `$y$` that is not predictable from `$x$`. In function estimation, we are interested in approximating `$f$` with a model or estimate `$\hat{f}$`. Function estimation is really just the same as estimating a parameter `$\theta$`; the function estimator `$\hat{f}$` is simply a point estimator in function space.

> **函数估计(Function Estimation)**: 有时我们会关注函数估计(或函数近似)。这时我们试图从输入向量 `$x$` 预测变量 `$y$`。我们假设有一个函数 `$f(x)$` 表示 `$y$` 和 `$x$` 之间的近似关系。

### Bias

The bias of an estimator is defined as:

`$$
bias(\hat{\theta}_{m})=E(\hat{\theta}_{m})-\theta \\
$$`

where the expectation is over the data (seen as samples from a random variable) and `$\theta$` is the true underlying value of `$\theta$` used to define the data generating distribution. An estimator `$\hat{\theta}_{m}$` is said to be **unbiased** if `$bias(\hat{\theta}_{m})=0$`, which implies that `$E[\hat{\theta}_{m}]=\theta$`. An estimator `$\hat{\theta}_{m}$` is said to be **asymptotically unbiased** if `$\lim_{m\to\infty}bias(\hat{\theta}_{m})=0$`, which implies that `$\lim_{m\to\infty}E[\hat{\theta}_{m}]=\theta$`.

> 如果 `$bias(\hat{\theta}_{m})=0$`，那么估计量 `$\hat{\theta}_{m}$` 被称为是 **无偏 (unbiased)** ，这意味着 `$E[\hat{\theta}_{m}]=\theta$`。如果 `$\lim_{m\to\infty}bias(\hat{\theta}_{m})=0$`，那么估计量 `$\hat{\theta}_{m}$` 被称为是 **渐近无偏(asymptotically unbiased)** ，这意味着 `$\lim_{m\to\infty}E[\hat{\theta}_{m}]=\theta$`。

### Variance and Standard Error

Another property of the estimator that we might want to consider is how much we expect it to vary as a function of the data sample. Just as we computed the expectation of the estimator to determine its bias, we can compute its variance. The variance of an estimator is simply the variance

`$$
Var(\hat{\theta}) \\
$$`

where the random variable is the training set. Alternately, the square root of the variance is called the **standard error**, denoted `$SE(\hat{\theta})$`.

The variance or the standard error of an estimator provides a measure of how we would expect the estimate we compute from data to vary as we independently resample the dataset from the underlying data generating process. Just as we might like an estimator to exhibit low bias we would also like it to have relatively low variance.

> 估计量的 **方差(variance)** 或 **标准差(standard error)** 告诉我们，当独立地从潜在的数据生成过程中重采样数据集时，如何期望估计的变化。正如我们希望估计的偏差较小，我们也希望其方差较小。

When we compute any statistic using a finite number of samples, our estimate of the true underlying parameter is uncertain, in the sense that we could have obtained other samples from the same distribution and their statistics would have been different. The expected degree of variation in any estimator is a source of error that we want to quantify.

The standard error of the mean is given by

`$$
SE(\hat{\mu}_{m})=\sqrt{Var \left[\frac{1}{m}\sum_{i=1}^{m}x^{(i)}\right]} = \frac{\sigma}{\sqrt{m}} \\
$$`

where `$\sigma^{2}$` is the true variance of the samples `$x^{i}$`. The standard error is often estimated by using an estimate of `$\sigma$`. Unfortunately, neither the square root of the sample variance nor the square root of the unbiased estimator of the variance provide an unbiased estimate of the standard deviation. Both approaches tend to underestimate the true standard deviation, but are still used in practice. The square root of the unbiased estimator of the variance is less of an underestimate. For large `$m$`, the approximation is quite reasonable.

The standard error of the mean is very useful in machine learning experiments. We often estimate the generalization error by computing the sample mean of the error on the test set. The number of examples in the test set determines the accuracy of this estimate. Taking advantage of the central limit theorem, which
tells us that the mean will be approximately distributed with a normal distribution, we can use the standard error to compute the probability that the true expectation falls in any chosen interval. For example, the 95% confidence interval centered on the mean `$\hat{\mu}_{m}$` is

`$$
(\hat{\mu}_{m}-1.96SE(\hat{\mu}_{m}),\hat{\mu}_{m}+1.96SE(\hat{\mu}_{m})) \\
$$`

under the normal distribution with mean `$\mu_{m}$` and variance `$SE(\hat{\mu}_{m})^{2}$`. In machine learning experiments, it is common to say that algorithm `$A$` is better than algorithm `$B$` if the upper bound of the 95% confidence interval for the error of algorithm `$A$` is less than the lower bound of the 95% confidence interval for the error of algorithm `$B$`.

### Trading off Bias and Variance to Minimize Mean Squared Error

Bias and variance measure two different sources of error in an estimator. Bias measures the expected deviation from the true value of the function or parameter. Variance on the other hand, provides a measure of the deviation from the expected estimator value that any particular sampling of the data is likely to cause.

What happens when we are given a choice between two estimators, one with more bias and one with more variance? How do we choose between them?

The most common way to negotiate this trade-off is to use cross-validation. Empirically, cross-validation is highly successful on many real-world tasks. Alternatively, we can also compare the **mean squared error (MSE)** of the estimates:

`$$
\begin{align*}
MSE &= E[(\hat{\theta}_{m}-\theta)^{2}] \\
&= Bias(\hat{\theta}_{m})^{2} + Var(\hat{\theta}_{m}) \\
\end{align*}
$$`

 The MSE measures the overall expected deviation—in a squared error sense— between the estimator and the true value of the parameter `$\theta$`.

 The relationship between bias and variance is tightly linked to the machine learning concepts of capacity, underfitting and overfitting. In the case where generalization error is measured by the MSE (where bias and variance are meaningful components of generalization error), increasing capacity tends to increase variance and decrease bias.

 <div align="center">
   <img src="/img_ML_Basics/ML_Basics_04_Bias_and_Variance.PNG" width=600px/>
 </div>
 <br>

 ### Consistency

 So far we have discussed the properties of various estimators for a training set of fixed size. Usually, we are also concerned with the behavior of an estimator as the amount of training data grows. In particular, we usually wish that, as the number of data points m in our dataset increases, our point estimates converge to the true value of the corresponding parameters. More formally, we would like that

 `$$
\mathrm{plim}_{m\to\infty}\hat{\theta}_{m}=\theta \\
 $$`

 The symbol `$\mathrm{plim}$` indicates convergence in probability, meaning that for any `$\epsilon >0$`, `$P(|\hat{\theta}_{m}-\theta|> \epsilon)\to 0$` as `$m \to \infty$`. The condition described by equation `$\mathrm{plim}_{m\to\infty}\hat{\theta}_{m}=\theta$` is known as **consistency**. It is sometimes referred to as weak consistency, with strong consistency referring to the **almost sure** convergence of `$\hat{\theta}$` to `$\theta$`. Almost sure convergence of a sequence of random variables `$x^{(1)},x^{(2)},\cdots$` to a value `$x$` occurs when `$p(\lim_{m\to\infty} x^{(m)} = x) = 1$`.

Consistency ensures that the bias induced by the estimator diminishes as the number of data examples grows. However, the reverse is not true—asymptotic unbiasedness does not imply consistency.

## Maximum Likelihood Estimation

We have seen some definitions of common estimators and analyzed their properties. But where did these estimators come from? Rather than guessing that some function might make a good estimator and then analyzing its bias and variance, we would like to have some principle from which we can derive specific functions that are good estimators for different models.

The most common such principle is the maximum likelihood principle.

Consider a set of m examples `$\mathbb{X}=x^{(1)},\cdots,x^{(m)}$` drawn independently from the true but unknown data generating distribution `$p_{data}(x)$`.

Let `$p_{model}(x;\theta)$` be a parametric family of probability distributions over the same space indexed by `$\theta$`. In other words, `$p_{model}(x;\theta)$` maps any configuration `$x$` to a real number estimating the true probability `$p_{data}(x)$`.

The maximum likelihood estimator for `$\theta$` is then defined as

`$$
\begin{align*}
\theta_{ML}&=\arg\max_{\theta}p_{model}(\mathbb{X};\theta) \\
&= \arg\max_{\theta}\prod_{i=1}^{m}p_{model}(x^{(i)};\theta) \\
\end{align*}
$$`

This product over many probabilities can be inconvenient for a variety of reasons. For example, it is prone to numerical underflow. To obtain a more convenient but equivalent optimization problem, we observe that taking the logarithm of the likelihood does not change its arg max but does conveniently transform a product into a sum:

`$$
\theta_{ML}=\arg\max_{\theta}\sum_{i=1}^{m}\log p_{model}(x^{(i)};\theta) \\
$$`

Because the `$\arg\max$` does not change when we rescale the cost function, we can divide by `$m$` to obtain a version of the criterion that is expressed as an expectation with respect to the empirical distribution `$\hat{p}_{data}$` defined by the training data:

`$$
\theta_{ML}=\arg\max_{\theta}\mathbb{E}_{x\sim \hat{p}_{data}}\log p_{model}(x;\theta) \\
$$`

### Conditional Log-Likelihood and Mean Squared Error

The maximum likelihood estimator can readily be generalized to the case where our goal is to estimate a conditional probability `$P(y|x;\theta)$` in order to predict `$y$` given `$x$`. This is actually the most common situation because it forms the basis for most supervised learning. If `$X$` represents all our inputs and `$Y$` all our observed targets, then the conditional maximum likelihood estimator is

`$$
\theta_{ML}=\arg\max_{\theta}P(Y|X;\theta) \\
$$`

If the examples are assumed to be i.i.d., then this can be decomposed into

`$$
\theta_{ML}=\arg\max_{\theta}\sum_{i=1}^{m}\log p_{model}(y^{(i)}|x^{(i)};\theta) \\
$$`

### Properties of Maximum Likelihood

The main appeal of the maximum likelihood estimator is that it can be shown to be the best estimator asymptotically, as the number of examples `$m\to\infty$`, in terms of its rate of convergence as `$m$` increases.

Under appropriate conditions, the maximum likelihood estimator has the property of consistency, meaning that as the number of training examples approaches infinity, the maximum likelihood estimate of a parameter converges to the true value of the parameter. These conditions are:

* The true distribution `$p_{data}$` must lie within the model family `$p_{model}(\cdot;\theta)$`. Otherwise, no estimator can recover `$p_{data}$` .
* The true distribution `$p_{data}$` must correspond to exactly one value of `$\theta$`. Other- wise, maximum likelihood can recover the correct `$p_{data}$` , but will not be able to determine which value of `$\theta$` was used by the data generating processing.

## Bayesian Statistics

We have discussed **frequentist statistics** and approaches based on estimating a single value of `$\theta$`, then making all predictions thereafter based on that one estimate. Another approach is to consider all possible values of `$\theta$` when making a prediction. The latter is the domain of **Bayesian statistics**.

The frequentist perspective is that the true parameter value `$\theta$` is fixed but unknown, while the point estimate `$\hat{\theta}$` is a random variable on account of it being a function of the dataset (which is seen as random).

The Bayesian perspective on statistics is quite different. The Bayesian uses probability to reflect degrees of certainty of states of knowledge. The dataset is directly observed and so is not random. On the other hand, the true parameter `$\theta$` is unknown or uncertain and thus is represented as a random variable.

> **频率派(frequentist statistics)** 的视角是真实参数 `$\theta$` 是未知的定值，而点估计 `$\hat{\theta}$` 是考虑数据集上函数(可以看作是随机的)的随机变量。 **贝叶斯统计(Bayesian statistics)** 的视角完全不同。贝叶斯用概率反映知识状态的确定性程度。数据 集能够被直接观测到，因此不是随机的。另一方面，真实参数 `$\theta$` 是未知或不确定的， 因此可以表示成随机变量。

Before observing the data, we represent our knowledge of `$\theta$` using the **prior probability distribution**, `$p(\theta)$` (sometimes referred to as simply “the prior”). Generally, the machine learning practitioner selects a prior distribution that is quite broad (i.e. with high entropy) to reflect a high degree of uncertainty in the value of `$\theta$` before observing any data. For example, one might assume a priori that `$\theta$` lies in some finite range or volume, with a uniform distribution. Many priors instead reflect a preference for “simpler” solutions (such as smaller magnitude coefficients, or a function that is closer to being constant).

Now consider that we have a set of data samples `$\{x^{(1)},\cdots,x^{(m)}\}$`. We can recover the effect of data on our belief about `$\theta$` by combining the data likelihood `$p(x^{(1)},\cdots,x^{(m)}|\theta)$` with the prior via Bayes’ rule:

`$$
p(\theta|x^{(1)},\cdots,x^{(m)})=\frac{p(x^{(1)},\cdots,x^{(m)}|\theta)p(\theta)}{p(x^{(1)},\cdots,x^{(m)})} \\
$$`

In the scenarios where Bayesian estimation is typically used, the prior begins as a relatively uniform or Gaussian distribution with high entropy, and the observation of the data usually causes the posterior to lose entropy and concentrate around a few highly likely values of the parameters.

Relative to maximum likelihood estimation, Bayesian estimation offers two important differences. First, unlike the maximum likelihood approach that makes predictions using a point estimate of `$\theta$`, the Bayesian approach is to make predictions using a full distribution over `$\theta$`. For example, after observing `$m$` examples, the predicted distribution over the next data sample, `$x^{(m+1)}$` , is given by

`$$
p(x^{(m+1)}|x^{(1)},\cdots,x^{(m)})=\int p(x^{(m+1)}|\ \theta)p(\theta \ |x^{(1)},\cdots,x^{(m)})d\theta  \\
$$`

Here each value of `$\theta$` with positive probability density contributes to the prediction of the next example, with the contribution weighted by the posterior density itself. After having observed `$\{x^{(1)},\cdots,x^{(m)}\}$`, if we are still quite uncertain about the value of `$\theta$`, then this uncertainty is incorporated directly into any predictions we might make.

The second important difference between the Bayesian approach to estimation and the maximum likelihood approach is due to the contribution of the Bayesian prior distribution. The prior has an influence by shifting probability mass density towards regions of the parameter space that are preferred a priori. In practice, the prior often expresses a preference for models that are simpler or more smooth. Critics of the Bayesian approach identify the prior as a source of subjective human judgment impacting the predictions.

Bayesian methods typically generalize much better when limited training data is available, but typically suffer from high computational cost when the number of training examples is large.

## Reference

[1]  Goodfellow, I., Bengio, Y., &amp; Courville, A. (2016, Nov 18). Deep Learning. https://www.deeplearningbook.org/contents/ml.html.
