---
title: "Monte Carlo Methods"
date: "2022-06-04"
tags: ["Monte Carlo Methods"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---

Monte Carlo methods require only ***experience***—sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. ***Learning from actual experience*** is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior. ***Learning from simulated experience*** is also powerful. Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form.

Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns. To ensure that well-defined returns are available, here ***we define Monte Carlo methods only for episodic tasks***. That is, we assume experience is divided into episodes, and that all episodes eventually terminate no matter what actions are selected. Only on the completion of an episode are value estimates and policies changed. Monte Carlo methods can thus be incremental in an episode-by-episode sense, but not in a step-by-step (online) sense.

### Monte Carlo Prediction

We begin by considering Monte Carlo methods for learning the state-value function for a given policy. Recall that the value of a state is the expected return—expected cumulative future discounted reward—starting from that state. An obvious way to estimate it from experience, then, is ***simply to average the returns observed*** after visits to that state. As more returns are observed, the average ***should converge to the expected value***. This idea underlies all Monte Carlo methods.

In particular, suppose we wish to estimate ``$v_{\pi}(s)$``, the value of a state ``$s$`` under policy ``$\pi$``, given a set of episodes obtained by following ``$\pi$`` and passing through ``$s$``. Each occurrence of state ``$s$`` in an episode is called a ***visit*** to ``$s$``. Of course, ``$s$`` may be visited multiple times in the same episode; let us call the first time it is visited in an episode the ***first visit*** to ``$s$``. The first-visit MC method estimates ``$v_{\pi}(s)$`` as the average of the returns following first visits to ``$s$``, whereas the ***every-visit*** MC method averages the returns following all visits to ``$s$``. These two Monte Carlo (MC) methods are very similar but have slightly different theoretical properties. First-visit MC is shown in procedural form in the box. Every-visit MC would be the same except without the check for ``$S_{t}$`` having occurred earlier in the episode.

<div align="center">
  <img src="/img_RL/05_MC_pred.PNG" width=650px/>
</div>
<br>

Both first-visit MC and every-visit MC converge to ``$v_{\pi}(s)$`` as the number of visits (or first visits) to ``$s$`` goes to infinity. This is easy to see for the case of first-visit MC. In this case each return is an independent, identically distributed estimate of ``$v_{\pi}(s)$`` with finite variance. By the law of large numbers the sequence of averages of these estimates converges to their expected value. Each average is itself an unbiased estimate, and the standard deviation of its error falls as ``$\frac{1}{\sqrt{n}}$``, where ``$n$`` is the number of returns averaged. Every-visit MC is less straightforward, but its estimates also converge quadratically to ``$v_{\pi}(s)$``.

### Monte Carlo Estimation of Action Values

If a model is not available, then it is particularly useful to estimate ***action values*** (the values of state–action pairs) rather than state values. With a model, ***state values*** alone are sufficient to determine a policy; one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state. Without a model, however, state values alone are not sufficient. One must explicitly estimate the value of each action in order for the values to be useful in suggesting a policy. ***Thus, one of our primary goals for Monte Carlo methods is to estimate ``$q_{*}$``***. To achieve this, we first consider the policy evaluation problem for action values.

The policy evaluation problem for action values is to estimate ``$q_{\pi}(s,a)$``, the expected return when starting in state ``$s$``, taking action ``$a$``, and thereafter following policy ``$\pi$``. The Monte Carlo methods for this are essentially the same as just presented for state values, except now we talk about visits to a state–action pair rather than to a state. A state–action pair ``$s$``, ``$a$`` is said to be visited in an episode if ever the state ``$s$`` is visited and action a is taken in it. *The every-visit MC method estimates the value of a state–action pair as the average of the returns that have followed all the visits to it*. The first-visit MC method averages the returns following the first time in each episode that the state was visited and the action was selected. These methods converge quadratically, as before, to the true expected values as the number of visits to each state–action pair approaches infinity.

The only complication is that ***many state–action pairs may never be visited***. If ``$\pi$`` is a deterministic policy, then in following ``$\pi$`` one will observe returns only for one of the actions from each state. With no returns to average, the Monte Carlo estimates of the other actions will not improve with experience. This is a serious problem because the purpose of learning action values is to help in choosing among the actions available in each state. ***To compare alternatives we need to estimate the value of all the actions from each state, not just the one we currently favor.***

This is the general problem of **maintaining exploration**. For policy evaluation to work for action values, we must assure continual exploration. One way to do this is by specifying that the episodes start in a state–action pair, and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. We call this the assumption of ***exploring starts***.

### Monte Carlo Control

We are now ready to consider how Monte Carlo estimation can be used in control, that is, to approximate optimal policies. In GPI one maintains both an approximate policy and an approximate value function. The value function is repeatedly altered to more closely approximate the value function for the current policy, and the policy is repeatedly improved with respect to the current value function. These two kinds of changes work against each other to some extent, as each creates a moving target for the other, but together ***they cause both policy and value function to approach optimality***.

<div align="center">
  <img src="/img_RL/05_MC_control.PNG" width=200px/>
</div>
<br>

To begin, let us consider a Monte Carlo version of classical policy iteration. In this method, we perform alternating complete steps of policy evaluation and policy improvement, beginning with an arbitrary policy ``$\pi_{0}$`` and ending with the optimal policy and optimal action-value function:

<div align="center">
  <img src="/img_RL/05_MC_control_progress.PNG" width=450px/>
</div>
<br>

where ``$\to^{E}$`` denotes a ***complete policy evaluation*** and ``$\to^{I}$`` denotes a ***complete policy improvement***. Policy evaluation is done exactly as described in the preceding section. Many episodes are experienced, with the approximate action-value function approaching the true function asymptotically. For the moment, let us assume that we do indeed observe an infinite number of episodes and that, in addition, the episodes are generated with exploring starts. Under these assumptions, the Monte Carlo methods will compute each ``$q_{\pi_{k}}$`` exactly, for arbitrary ``$\pi_{k}$``.

Policy improvement is done by making the policy ***greedy*** with respect to the current value function. In this case we have an action-value function, and therefore no model is needed to construct the greedy policy. For any action-value function ``$q$``, the corresponding greedy policy is the one that, for each ``$s\in\mathcal{S}$``, deterministically chooses an action with maximal action-value:

`$$
\pi(s)\doteq \arg\max_{a}q(s,a) \\
$$`

Policy improvement then can be done by constructing each ``$\pi_{k+1}$`` as the greedy policy with respect to ``$q_{\pi_{k}}$``. The policy improvement theorem then applies to ``$\pi_{k}$`` and ``$\pi_{k+1}$`` because, for all ``$s\in\mathcal{S}$``,

`$$
\begin{align}
q_{\pi_{k}}(s,\pi_{k+1}(s))&=q_{\pi_{k}}(s,\arg\max_{a}q_{\pi_{k}}(s,a)) \\
&= \max_{a}q_{\pi_{k}}(s,a)\\
&\geq q_{\pi_{k}}(s,\pi_{k}(s)) \\
&\geq v_{\pi_{k}}(s)
\end{align}
$$`

The theorem assures us that each ``$\pi_{k+1}$`` is uniformly better than ``$\pi_{k}$``, or just as good as ``$\pi_{k}$``, in which case they are both optimal policies. This in turn assures us that the overall process converges to the optimal policy and optimal value function. In this way Monte Carlo methods can be used to find optimal policies given only sample episodes and no other knowledge of the environment’s dynamics.

For Monte Carlo policy iteration it is natural to alternate between evaluation and improvement on an episode-by-episode basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode. A complete simple algorithm along these lines, which we call ***Monte Carlo ES***, for Monte Carlo with Exploring Starts, is given in pseudocode in the box on the next page.

<div align="center">
  <img src="/img_RL/05_MCES.PNG" width=650px/>
</div>
<br>

### Monte Carlo Control without Exploring Starts

How can we avoid the unlikely assumption of exploring starts? The only general way to ensure that all actions are selected infinitely often is for the agent to continue to select them. There are two approaches to ensuring this, resulting in what we call ***on-policy methods*** and ***off-policy methods***. On-policy methods attempt to evaluate or improve the policy that is used to make decisions, whereas off-policy methods evaluate or improve a policy different from that used to generate the data. The Monte Carlo ES method developed above is an example of an on-policy method.

In on-policy control methods the policy is generally soft, meaning that ``$\pi(a|s)>0$`` for all ``$s \in \mathcal{S}$`` and all ``$a \in \mathcal{A}(s)$``, but gradually shifted closer and closer to a deterministic optimal policy. The on-policy method we present in this section uses ``$\varepsilon$``-greedy policies, meaning that most of the time they choose an action that has maximal estimated action value, but with probability ``$\varepsilon$`` they instead select an action at random. That is, all non-greedy actions are given the minimal probability of selection, ``$\frac{\varepsilon}{|\mathcal{A}(s)|}$``, and the remaining bulk of the probability, ``$1- \varepsilon+ \frac{\varepsilon}{|\mathcal{A}(s)|}$`` , is given to the greedy action. The ``$\varepsilon$``-greedy policies are examples of ``$\varepsilon$``-soft policies, defined as policies for which ``$\pi(a|s)\geq\frac{\varepsilon}{|\mathcal{A}(s)|}$`` for all states and actions, for some ``$\varepsilon$`` > 0. Among ``$\varepsilon$``-soft policies, ``$\varepsilon$``-greedy policies are in some sense those that are closest to greedy.

The overall idea of on-policy Monte Carlo control is still that of GPI. As in Monte Carlo ES, we use first-visit MC methods to estimate the action-value function for the current policy. Without the assumption of exploring starts, however, we cannot simply improve the policy by making it greedy with respect to the current value function, because that would prevent further exploration of non-greedy actions. Fortunately, GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a greedy policy. In our on-policy method we will move it only to an ``$\varepsilon$``-greedy policy. ***For any ``$\varepsilon$``-soft policy, ``$\pi$``, any ``$\varepsilon$``-greedy policy with respect to ``$q_{\pi}$`` is guaranteed to be better than or equal to ``$\pi$``.*** The complete algorithm is given in the box below.

<div align="center">
  <img src="/img_RL/05_on_policy_first.PNG" width=650px/>
</div>
<br>

### Incremental Implementation

Suppose we have a sequence of returns ``$G_{1},G_{2},\cdots,G_{n-1}$``, all starting in the same state and each with a corresponding random weight ``$W_{i}$`` (e.g., ``$W_{i}=\rho_{t_{i}:T(t_{i})-1}$``). We wish to form the estimate

`$$
V_{n}\doteq\frac{\sum_{k=1}^{n-1}W_{k}G_{k}}{\sum_{k=1}^{n-1}W_{k}}, \ \ n \geq2 \\
$$`

and keep it up-to-date as we obtain a single additional return ``$G_{n}$``. In addition to keeping track of ``$V_{n}$``, we must maintain for each state the cumulative sum ``$C_{n}$`` of the weights given to the first ``$n$`` returns. The update rule for ``$V_{n}$`` is

`$$
V_{n+1}\doteq V_{n}+\frac{W_{n}}{C_{n}}[G_{n}-V_{n}], \ \ n \geq1 \\
$$`

and

`$$
C_{n+1}\doteq C_{n}+W_{n+1} \\
$$`

where ``$C_{0}\doteq 0$`` (and ``$V_{1}$`` is arbitrary and thus need not be specified). The box below contains a complete episode-by-episode incremental algorithm for Monte Carlo policy evaluation.

<div align="center">
  <img src="/img_RL/05_off_MC_pred.PNG" width=650px/>
</div>
<br>

### Off-policy Monte Carlo Control

We are now ready to present an example of the second class of learning control methods we consider in this book: off-policy methods. Recall that the distinguishing feature of on-policy methods is that they estimate the value of a policy while using it for control. In off-policy methods these two functions are separated. The policy used to generate behavior, called the ***behavior policy***, may in fact be unrelated to the policy that is evaluated and improved, called the ***target policy***. An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions.

Off-policy Monte Carlo control methods use one of the techniques presented in the preceding two sections. They follow the behavior policy while learning about and improving the target policy. These techniques require that the behavior policy has a nonzero probability of selecting all actions that might be selected by the target policy (coverage). To explore all possibilities, we require that the behavior policy be soft (i.e., that it select all actions in all states with nonzero probability). The box below shows an off-policy Monte Carlo control method, based on GPI and weighted importance sampling, for estimating ``$\pi_{*}$`` and ``$q_{*}$``.

<div align="center">
  <img src="/img_RL/05_off_MC_alg.PNG" width=650px/>
</div>
<br>

### Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
