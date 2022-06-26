---
title: "Temporal-Difference Learning"
date: "2022-06-11"
tags: ["TD Learning"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---

If one had to identify one idea as central and novel to reinforcement learning, it would undoubtedly be ***temporal-difference (TD)*** learning. TD learning is a combination of Monte Carlo ideas and dynamic programming (DP) ideas. Like Monte Carlo methods, TD methods can **learn directly from raw experience without a model of the environment’s dynamics**. Like DP, TD methods update estimates based in part on other learned estimates, without waiting for a final outcome (they bootstrap). The relationship between TD, DP, and Monte Carlo methods is a recurring theme in the theory of reinforcement learning

### TD Prediction

Both TD and Monte Carlo methods use experience to solve the prediction problem. Given some experience following a policy ``$\pi$``, both methods update their estimate ``$V$`` of ``$v_{\pi}$`` for the nonterminal states ``$S_{t}$`` occurring in that experience. Roughly speaking, Monte Carlo methods wait until the return following the visit is known, then use that return as a target for ``$V(S_{t})$``. A simple every-visit Monte Carlo method suitable for non-stationary environments is

`$$
V(S_{t})\leftarrow V(S_{t})+\alpha[G_{t}-V(S_{t})] \\
$$`

where ``$G_{t}$`` is the actual return following time ``$t$``, and ``$\alpha$`` is a constant step-size parameter. Let us call this method constant-``$\alpha$`` MC. Whereas Monte Carlo methods must wait until the end of the episode to determine the increment to ``$V(S_{t})$`` (only then is ``$G_{t}$`` known), TD methods need to wait only until the next time step. At time ``$t+1$`` they immediately form a target and make a useful update using the observed reward ``$R_{t+1}$`` and the estimate ``$V(S_{t+1})$``. The simplest TD method makes the update

`$$
V(S_{t})\leftarrow V(S_{t})+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_{t})] \\
$$`

immediately on transition to ``$S_{t+1}$`` and receiving ``$R_{t+1}$``. In effect, the target for the Monte Carlo update is ``$G_{t}$``, whereas the target for the TD update is ``$R_{t+1}+\gamma V(S_{t+1})$``. This TD method is called TD(0), or one-step TD.  The box below specifies TD(0) completely in procedural form.

<div align="center">
  <img src="/img_RL/06_TD_est.PNG" width=650px/>
</div>
<br>

Because TD(0) bases its update in part on an existing estimate, we say that it is a bootstrapping method. Note that the quantity in brackets in the TD(0) update is a sort of error, measuring the difference between the estimated value of ``$S_{t}$`` and the better estimate ``$R_{t+1}+\gamma V(S_{t+1})$``. This quantity, called the TD error, arises in various forms throughout reinforcement learning:

`$$
\delta_{t} \doteq R_{t+1}+\gamma V(S_{t+1}) -V(S_{t}) \\
$$`

Notice that the TD error at each time is the error in the estimate made at that time. Because the TD error depends on the next state and next reward, it is not actually available until one time step later. That is, ``$\delta_{t}$`` is the error in ``$V(S_{t})$``, available at time ``$t+1$``. Also note that if the array ``$V$`` does not change during the episode (as it does not in Monte Carlo methods), then the Monte Carlo error can be written as a sum of TD errors:

`$$
\begin{align}
G_{t}-V(S_{t}) &=  R_{t+1}+\gamma V(S_{t+1}) -V(S_{t}) + \gamma V(S_{t+1}-\gamma V(S_{t+1}) \\
&=\delta_{t}+\gamma(G_{t+1}-V(S_{t+1})) \\
&=\delta_{t}+\gamma\delta_{t+1}+\gamma^{2}(G_{t+2}-V(S_{t+2})) \\
&= \delta_{t}+\gamma\delta_{t+1}+\gamma^{2}\delta_{t+2}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}(G_{T}-V(S_{T})) \\
&= \delta_{t}+\gamma\delta_{t+1}+\gamma^{2}\delta_{t+2}+\cdots+\gamma^{T-t-1}\delta_{T-1}+\gamma^{T-t}(0-0) \\
&= \sum^{T-1}_{k=t}\gamma^{k-t}\delta_{k}
\end{align}
$$`

This identity is not exact if ``$V$`` is updated during the episode (as it is in TD(0)), but if the step size is small then it may still hold approximately. Generalizations of this identity play an important role in the theory and algorithms of temporal-difference learning.

### Advantages of TD Prediction Methods

TD methods update their estimates based in part on other estimates. They learn a guess from a guess—they bootstrap. Is this a good thing to do? Obviously, TD methods have an advantage over DP methods in that ***they do not require a model of the environment, of its reward and next-state probability distributions***.

***The next most obvious advantage of TD methods over Monte Carlo methods is that they are naturally implemented in an online, fully incremental fashion.*** With Monte Carlo methods one must wait until the end of an episode, because only then is the return known, whereas with TD methods one need wait only one time step. Surprisingly often this turns out to be a critical consideration. Some applications have very long episodes, so that delaying all learning until the end of the episode is too slow. Other applications are continuing tasks and have no episodes at all. Finally, as we noted in the previous chapter, some Monte Carlo methods must ignore or discount episodes on which experimental actions are taken, which can greatly slow learning. TD methods are much less susceptible to these problems because they learn from each transition regardless of what subsequent actions are taken.

If both TD and Monte Carlo methods converge asymptotically to the correct predictions, then a natural next question is “Which gets there first?” In other words, which method learns faster? Which makes the more efficient use of limited data? At the current time this is an open question in the sense that no one has been able to prove mathematically that one method converges faster than the other. In fact, it is not even clear what is the most appropriate formal way to phrase this question! In practice, however, ***TD methods have usually been found to converge faster than constant-``$\alpha$`` MC methods on stochastic tasks***.

### Sarsa: On-policy TD Control

We turn now to the use of TD prediction methods for the control problem. As usual, we follow the pattern of generalized policy iteration (GPI), only this time using TD methods for the evaluation or prediction part. As with Monte Carlo methods, we face the need to trade off exploration and exploitation, and again approaches fall into two main classes: on-policy and off-policy.

The first step is to learn an action-value function rather than a state-value function. In particular, for an on-policy method we must estimate ``$q_{\pi}(s,a)$`` for the current behavior policy ``$\pi$`` and for all states ``$s$`` and actions ``$a$``. This can be done using essentially the same TD method described above for learning ``$v_{\pi}$``. Recall that an episode consists of an alternating sequence of states and state–action pairs:

<div align="center">
  <img src="/img_RL/06_on_policy_sarsa_progress.PNG" width=550px/>
</div>
<br>

In the previous section we considered transitions from state to state and learned the values of states. Now we consider transitions from state–action pair to state–action pair, and learn the values of state–action pairs. Formally these cases are identical: they are both Markov chains with a reward process. The theorems assuring the convergence of state values under TD(0) also apply to the corresponding algorithm for action values:

`$$
Q(S_{t},A_{t})\leftarrow Q(S_{t},A_{t})+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_{t},A_{t})] \\
$$`

This update is done after every transition from a nonterminal state ``$S_{t}$``. If ``$S_{t+1}$`` is terminal, then ``$Q(S_{t+1},A_{t+1})$`` is defined as zero. This rule uses every element of the quintuple of events, ``$(S_{t},A_{t},R_{t+1},S_{t+1},A_{t+1})$``, that make up a transition from one state–action pair to the next. This quintuple gives rise to the name Sarsa for the algorithm. The backup diagram for Sarsa is as shown to the right.

It is straightforward to design an on-policy control algorithm based on the Sarsa prediction method. As in all on-policy methods, we continually estimate ``$q_{\pi}$`` for the behavior policy ``$\pi$``, and at the same time change ``$\pi$`` toward greediness with respect to ``$q_{\pi}$``. The general form of the Sarsa control algorithm is given in the box on the next page. The convergence properties of the Sarsa algorithm depend on the nature of the policy’s dependence on ``$Q$``. For example, one could use "``$\varepsilon$``-greedy or "``$\varepsilon$``-soft policies.

<div align="center">
  <img src="/img_RL/06_On_policy_sarsa_alg.PNG" width=650px/>
</div>
<br>

### ``$Q$``-learning: Off-policy TD Control

One of the early breakthroughs in reinforcement learning was the development of an off-policy TD control algorithm known as ``$Q$``-learning (Watkins, 1989), defined by

`$$
Q(S_{t},A_{t})\leftarrow Q(S_{t},A_{t})+\alpha[R_{t+1}+\gamma \max_{a}Q(S_{t+1},a)-Q(S_{t},A_{t})] \\
$$`

In this case, the learned action-value function, ``$Q$``, directly approximates ``$q_{*}$``, the optimal action-value function, independent of the policy being followed. This dramatically simplifies the analysis of the algorithm and enabled early convergence proofs. The policy still has an effect in that it determines which state–action pairs are visited and updated. However, all that is required for correct convergence is that all pairs continue to be updated. This is a minimal requirement in the sense that any method guaranteed to find optimal behavior in the general case must require it. Under this assumption and a variant of the usual stochastic approximation conditions on the sequence of step-size parameters, ``$Q$`` has been shown to converge with probability 1 to ``$q_{*}$``. The ``$Q$``-learning algorithm is shown below in procedural form.

<div align="center">
  <img src="/img_RL/06_QLearning_alg.PNG" width=650px/>
</div>
<br>

### Expected Sarsa

Consider the learning algorithm that is just like ``$Q$``-learning except that instead of the maximum over next state–action pairs it uses the expected value, taking into account how likely each action is under the current policy. That is, consider the algorithm with the update rule

`$$
\begin{align}
Q(S_{t},A_{t})&\leftarrow Q(S_{t},A_{t})+\alpha[R_{t+1}+\gamma \mathbb{E}_{\pi}[Q(S_{t+1},A_{t+1})|S_{t+1}]-Q(S_{t},A_{t})] \\
&\leftarrow Q(S_{t},A_{t})+\alpha[R_{t+1}+\gamma\sum_{a}\pi(a|S_{t+1})Q(S_{t+1},a) -Q(S_{t},A_{t})] \\
\end{align}
$$`

but that otherwise follows the schema of ``$Q$``-learning. Given the next state, ``$S_{t+1}$``, this algorithm moves deterministiacally in the same direction as Sarsa moves in expectation, and accordingly it is called Expected Sarsa.

<div align="center">
  <img src="/img_RL/06_exp_sarsa.PNG" width=450px/>
</div>
<br>

### Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
