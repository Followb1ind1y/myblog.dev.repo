---
title: "Dynamic Programming"
date: "2022-05-28"
tags: ["Dynamic Programming"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---

The term ***dynamic programming*** (DP) refers to a collection of algorithms that can be used to **compute optimal policies** given a perfect model of the environment as a Markov decision process (MDP). Classical DP algorithms are of limited utility in reinforcement learning both because of their assumption of a perfect model and because of their great computational expense, but they are still important theoretically.

Starting with this chapter, we usually assume that the environment is a finite MDP. That is, we assume that its state, action, and reward sets, ``$\mathcal{S}$``, ``$\mathcal{A}$``, and ``$\mathcal{R}$`` are finite, and that its dynamics are given by a set of probabilities ``$p(s',r|s,a)$``, for all ``$s \in \mathcal{S}$``, ``$a \in \mathcal{A}(s)$``, ``$r \in \mathcal{R}$``, and ``$s'\in\mathcal{S}^{+}$`` (``$S^{+}$`` is ``$\mathcal{S}$`` plus a terminal state if the problem is episodic). Although DP ideas can be applied to problems with continuous state and action spaces, exact solutions are possible only in special cases. A common way of obtaining approximate solutions for tasks with continuous states and actions is to quantize the state and action spaces and then apply finite-state DP methods.

The key idea of DP, and of reinforcement learning generally, ***is the use of value functions to organize and structure the search for good policies.*** We know that we can easily obtain optimal policies once we have found the optimal value functions, ``$v_{*}$ or $q_{*}$``, which satisfy the Bellman optimality equations:

`$$
v_{*}(s,a)=\max_{a}\mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})|S_{t}=s, A_{t}=a] \\
=\max_{a}\sum_{s',r}p(s',r|s,a)[r+ \gamma v_{*}(s')] \\
$$`

or

`$$
q_{*}(s,a)=\mathbb{E}[R_{t+1}+\gamma \max_{a'}q_{*}(S_{t+1},a')|S_{t}=s, A_{t}=a] \\
=\sum_{s',r}p(s',r|s,a)[r+ \gamma \max_{a'}q_{*}(s',a')] \\
$$`

all ``$s \in \mathcal{S}$``, ``$a \in \mathcal{A}(s)$``, and ``$s'\in\mathcal{S}^{+}$``. As we shall see, DP algorithms are obtained by turning Bellman equations such as these into assignments, that is, into update rules for improving approximations of the desired value functions.

### Policy Evaluation (Prediction)

First we consider how to compute the state-value function ``$v_{\pi}$`` for an arbitrary policy ``$\pi$``. This is called ***policy evaluation*** in the DP literature. We also refer to it as the ***prediction problem***. For all ``$s \in \mathcal{S}$``,

`$$
\begin{align}
v_{\pi}(s)&\doteq\mathbb{E}[G_{t}|S_{t}=s] \\
&=\mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_{t}=s] \\
&= \sum_{a}\pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)[r+\gamma\mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']] \\
&= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')] \\
\end{align}
$$`

where ``$\pi(a|s)$`` is the probability of taking action ``$a$`` in state ``$s$`` under policy ``$\pi$``, and the expectations are subscripted by ``$\pi$`` to indicate that they are conditional on ``$\pi$`` being followed. The existence and uniqueness of ``$v_{\pi}$`` are guaranteed as long as either ``$\gamma <1$`` or eventual termination is guaranteed from all states under the policy ``$\pi$``.

If the environment’s dynamics are completely known, then the previous equation is a system of ``$|\mathcal{S}|$`` simultaneous linear equations in ``$|\mathcal{S}|$`` unknowns (the ``$v_{\pi}(s)$``, ``$s \in \mathcal{S}$``). In principle, its solution is a straightforward, if tedious, computation. For our purposes, iterative solution methods are most suitable. Consider a sequence of approximate value functions ``$v_{0},v_{1},v_{2}.\cdots,$`` each mapping ``$\mathcal{S}^{+}$`` to ``$\mathbb{R}$`` (the real numbers). The initial approximation, ``$v_{0}$``, is chosen arbitrarily (except that the terminal state, if any, must be given value 0), and each successive approximation is obtained by using the Bellman equation for ``$v_{\pi}$`` as an update rule:

`$$
\begin{align}
v_{k+1}(s)
&=\mathbb{E}_{\pi}[R_{t+1}+\gamma v_{k}S_{t+1}|S_{t}=s] \\
&= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{k}(s')] \\
\end{align}
$$`

for all ``$s\in \mathcal{S}$``. Clearly, ``$v_{k}=v_{\pi}$`` is a fixed point for this update rule because the Bellman equation for ``$v_{\pi}$`` assures us of equality in this case. Indeed, the sequence {``$v_{k}$``} can be shown in general to converge to ``$v_{\pi}$`` as ``$k \to \infty$`` under the same conditions that guarantee the existence of ``$v_{\pi}$``. This algorithm is called iterative policy evaluation.

<div align="center">
  <img src="/img_RL/04_policy_evl_alg.PNG" width=650px/>
</div>
<br>

<div align="center">
  <img src="/img_RL/04_Vs_exp.PNG" width=650px/>
</div>
<br>

### Policy Improvement

Our reason for computing the value function for a policy is to help find better policies. Suppose we have determined the value function ``$v_{\pi}$`` for an arbitrary deterministic policy ``$\pi$``. For some state ``$s$`` we would like to know whether or not we should change the policy to deterministically choose an action ``$a \neq \pi(s)$``. We know how good it is to follow the current policy from ``$s$``— that is ``$v_{\pi}(s)$`` — but would it be better or worse to change to the new policy? One way to answer this question is to consider selecting a in ``$s$`` and thereafter following the existing policy, ``$\pi$``. The value of this way of behaving is

`$$
q_{\pi}(s,a)=\mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s, A_{t}=a] \\
=\sum_{s',r}p(s',r|s,a)[r+ \gamma v_{\pi}(s')] \\
$$`

The key criterion is whether this is greater than or less than ``$v_{\pi}(s)$``. If it is greater — that is, if it is better to select ``$a$`` once in ``$s$`` and thereafter follow ``$\pi$`` than it would be to follow ``$\pi$`` all the time—then one would expect it to be better still to select a every time ``$s$`` is encountered, and that the new policy would in fact be a better one overall.

That this is true is a special case of a general result called the ***policy improvement*** theorem. Let ``$\pi$`` and ``$\pi'$`` be any pair of deterministic policies such that, for all ``$s\in \mathcal{S}$``,

`$$
q_{\pi}(s,\pi'(s)) \geq v_{\pi}(s) \\
$$`

Then the policy ``$\pi'$`` must be as good as, or better than, ``$\pi$``. That is, it must obtain greater or equal expected return from all states ``$s \in\mathcal{S}$``:

`$$
v_{\pi'}(s) \geq v_{\pi}(s) \\
$$`

So far we have seen how, given a policy and its value function, we can easily evaluate a change in the policy at a single state. It is a natural extension to consider changes at all states, selecting at each state the action that appears best according to ``$q_{\pi}(s,a)$``. In other words, to consider the new greedy policy, ``$\pi'$``, given by

`$$
\begin{align}
\pi'(s)&\doteq \arg\max_{a}q_{\pi}(s,a) \\
&= \arg\max_{a} \mathbb{E}[R_{t+1}+\gamma v_{\pi}(S_{t+1})|S_{t}=s, A_{t}=a] \\
&= \arg\max_{a}\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')] \\
\end{align}
$$`

where ``$\arg\max_{a}$`` denotes the value of ``$a$`` at which the expression that follows is maximized (with ties broken arbitrarily). The greedy policy takes the action that looks best in the short term—after one step of lookahead—according to ``$v_{\pi}$``. By construction, the greedy policy meets the conditions of the ***policy improvement theorem***, so we know that it is as good as, or better than, the original policy. The process of making a new policy that improves on an original policy, by making it greedy with respect to the value function of the original policy, is called ***policy improvement***.

Suppose the new greedy policy, ``$\pi'$``, is as good as, but not better than, the old policy ``$\pi$``. Then ``$v_{\pi}=v_{\pi'}$`` , and it follows that for all ``$s \in \mathcal{S}$``:

`$$
\begin{align}
v_{\pi'}(s)&= \max_{a} \mathbb{E}[R_{t+1}+\gamma v_{\pi'}(S_{t+1})|S_{t}=s, A_{t}=a] \\
&= \max_{a}\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')] \\
\end{align}
$$`

But this is the same as the Bellman optimality equation, and therefore, ``$v_{\pi'}$`` must be ``$v_{*}$``, and both ``$\pi$`` and ``$\pi'$`` must be optimal policies. Policy improvement thus must give us a strictly better policy except when the original policy is already optimal.

### Policy Iteration

Once a policy, ``$\pi$``, has been improved using ``$v_{\pi}$`` to yield a better policy, ``$\pi'$``, we can then compute ``$v_{\pi'}$`` and improve it again to yield an even better ``$\pi''$``. We can thus obtain a sequence of monotonically improving policies and value functions:

<div align="center">
  <img src="/img_RL/04_Iter.PNG" width=500px/>
</div>
<br>

where ``$\to^{E}$`` denotes a ***policy evaluation*** and ``$\to^{I}$`` denotes a ***policy improvement***. Each policy is guaranteed to be a strict improvement over the previous one (unless it is already optimal). Because a finite MDP has only a finite number of policies, this process must converge to an optimal policy and the optimal value function in a finite number of iterations.

This way of finding an optimal policy is called ***policy iteration***. Note that each policy evaluation, itself an iterative computation, is started with the value function for the previous policy. This typically results in a great increase in the speed of convergence of policy evaluation (presumably because the value function changes little from one policy to the next).

<div align="center">
  <img src="/img_RL/04_Iter_alg.PNG" width=650px/>
</div>
<br>

### Value Iteration

One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set. If policy evaluation is done iteratively, then convergence exactly to ``$v_{\pi}$`` occurs only in the limit. Must we wait for exact convergence, or can we stop short of that?

In fact, the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state). This algorithm is called ***value iteration***. It can be written as a particularly simple update operation that combines the policy improvement and truncated policy evaluation steps:

`$$
\begin{align}
v_{k+1}(s)&\doteq \max_{a} \mathbb{E}[R_{t+1}+\gamma v_{k}(S_{t+1})|S_{t}=s, A_{t}=a] \\
&= \max_{a}\sum_{s',r}p(s',r|s,a)[r+\gamma v_{k}(s')] \\
\end{align}
$$`

for all ``$s \in \mathcal{S}$``. For arbitrary ``$v_{0}$``, the sequence ``$\{v_{k}\}$`` can be shown to converge to ``$v_{*}$`` under the same conditions that guarantee the existence of ``$v_{*}$``.

Another way of understanding value iteration is by reference to the Bellman optimality equation. Note that value iteration is obtained simply by turning the Bellman optimality equation into an update rule. Also note how the value iteration update is identical to the policy evaluation update except that it requires the maximum to be taken over all actions. Another way of seeing this close relationship is to compare the backup diagrams for these algorithms (policy evaluation) and (value iteration). These two are the natural backup operations for computing ``$v_{\pi}$`` and ``$v_{*}$``.

Finally, let us consider how value iteration terminates. Like policy evaluation, value iteration formally requires an infinite number of iterations to converge exactly to ``$v_{*}$``. In practice, we stop once the value function changes by only a small amount in a sweep. The box below shows a complete algorithm with this kind of termination condition.

<div align="center">
  <img src="/img_RL/04_Value_iter.PNG" width=650px/>
</div>
<br>

Value iteration effectively combines, in each of its sweeps, one sweep of policy evaluation and one sweep of policy improvement. Faster convergence is often achieved by interposing multiple policy evaluation sweeps between each policy improvement sweep. In general, the entire class of truncated policy iteration algorithms can be thought of as sequences of sweeps, some of which use policy evaluation updates and some of which use value iteration updates. Because the max operation is the only difference between these updates, this just means that the max operation is added to some sweeps of policy evaluation. All of these algorithms converge to an optimal policy for discounted finite MDPs.

### Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
