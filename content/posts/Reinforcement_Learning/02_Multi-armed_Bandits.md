---
title: "Multi-armed Bandits"
date: "2022-05-21"
tags: ["Multi-armed Bandits"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---

Consider the following learning problem. You are faced repeatedly with a choice among $k$ different options, or actions. After each choice you receive a numerical reward chosen from a stationary probability distribution that depends on the action you selected. Your objective is to ***maximize the expected total reward over some time period***.

This is the original form of the ***`$k$`-armed bandit problem***, so named by analogy to a slot machine, or “one-armed bandit,” except that it has `$k$` levers instead of one. Each action selection is like a play of one of the slot machine’s levers, and the rewards are the payoffs for hitting the jackpot. Through repeated action selections you are to maximize your winnings by concentrating your actions on the best levers.

In our `$k$`-armed bandit problem, each of the `$k$` actions has an expected or mean reward given that that action is selected; let us call this the **value of that action**. We denote the action selected on time step `$t$`  as `$A_{t}$`, and the corresponding reward as `$R_{t}$`. The value then of an arbitrary action `$a$`, denoted `$q_{*}(a)$`, is the expected reward given that `$a$` is selected:

`$$
q_{*}(a)\doteq \mathbb{E}[R_{t}|A_{t}=a] \\
$$`

If you knew the value of each action, then it would be trivial to solve the `$k$`-armed bandit problem: you would always select the action with highest value. We assume that you do not know the action values with certainty, although you may have estimates. We denote the estimated value of action a at time step `$t$` as `$Q_{t}(a)$`. We would like `$Q_{t}(a)$` to be close to `$q_{*}(a)$`.

If you maintain estimates of the action values, then at any time step there is at least one action whose estimated value is greatest. We call these the ***greedy*** actions. When you select one of these actions, we say that you are ***exploiting*** your current knowledge of the values of the actions. If instead you select one of the ***non-greedy*** actions, then we say you are ***exploring***, because this enables you to improve your estimate of the non-greedy action’s value. Exploitation is the right thing to do to maximize the expected reward on the one step, but exploration may produce the greater total reward in the long run. If you have many time steps ahead on which to make action selections, then it may be better to explore the non-greedy actions and discover which of them are better than the greedy action. Reward is lower in the short run, during exploration, but higher in the long run because after you have discovered the better actions, you can exploit them many times. Because it is not possible both to explore and to exploit with any single action selection, one often refers to the “conflict” between exploration and exploitation.

### Incremental Implementation

We now turn to the question of how these averages can be computed in a computationally efficient manner, in particular, with constant memory and constant per-time-step computation.

To simplify notation we concentrate on a single action. Let `$R_{i}$` now denote the reward received after the `$i$`-th selection of **this action**, and let `$Q_{n}$` denote the estimate of its action value after it has been selected `$n-1$` times, which we can now write simply as

`$$
Q_{n}\doteq \frac{R_{1}+R_{2}+\cdots+R_{n-1}}{n-1}
$$`

The obvious implementation would be to maintain a record of all the rewards and then perform this computation whenever the estimated value was needed. However, if this is done, then the memory and computational requirements would grow over time as more rewards are seen. Each additional reward would require additional memory to store it and additional computation to compute the sum in the numerator.

As you might suspect, this is not really necessary. It is easy to devise incremental formulas for updating averages with small, constant computation required to process each new reward. Given `$Q_{n}$` and the `$n$`-th reward,  `$R_{n}$`, the new average of all `$n$` rewards can be computed by

`$$
\begin{align}Q_{n+1} &= \frac{1}{n}\sum_{i=1}^{n}R_{i} \\
&= \frac{1}{n}(R_{n}+\sum_{i=1}^{n-1}R_{i}) \\
&= \frac{1}{n}(R_{n}+(n-1)\frac{1}{n-1}\sum_{i=1}^{n-1}R_{i}) \\
&=\frac{1}{n}(R_{n}+(n-1)Q_{n})\\
&=\frac{1}{n}(R_{n}+nQ_{n}-Q_{n})\\
&=Q_{n}+\frac{1}{n}[R_{n}-Q_{n}]
\end{align}
$$`

which holds even for `$n=1$`, obtaining `$Q_{2}=R_{1}$` for arbitrary `$Q_{1}$`. This implementation requires memory only for `$Q_{n}$` and `$n$`, and only the small computation for each new reward. The general form of this equation is:

`$$
\mathrm{NewEstimate} \leftarrow \mathrm{OldEstimate} + \mathrm{StepSize}[\mathrm{Target}- \mathrm{OldEstimate}]
$$`

The expression `$[\mathrm{Target}- \mathrm{OldEstimate}]$` is an **error** in the estimate. It is reduced by taking a step toward the “Target.” The target is presumed to indicate a desirable direction in which to move, though it may be noisy.

<div align="center">
  <img src="/img_RL/02_Band_alg.PNG" width=650px/>
</div>

### Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate for stationary bandit problems, that is, for bandit problems in which the reward probabilities do not change over time. As noted earlier, we often encounter reinforcement learning problems that are effectively non-stationary. In such cases it makes sense to give more weight to recent rewards than to long-past rewards. One of the most popular ways of doing this is to use a constant step-size parameter. For example, the incremental update rule above for updating an average `$Q_{n}$` of the `$n-1$` past rewards is modified to be

`$$
Q_{n+1}\doteq Q_{n}+\alpha[R_{n}-Q_{n}]
$$`

where the step-size parameter `$\alpha \in (0,1]$` is constant. This results in `$Q_{n+1}$` being a weighted average of past rewards and the initial estimate `$Q_{1}$`:

`$$
\begin{align}Q_{n+1}&= Q_{n}+\alpha[R_{n}-Q_{n}] \\
&= (1-\alpha)^{n}Q_{1}+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}R_{i}
\end{align}
$$`

We call this a **weighted average** because the sum of the weights is

`$$
(1-\alpha)^{n}+\sum_{i=1}^{n}\alpha(1-\alpha)^{n-i}=1 \\
$$`.


### Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
