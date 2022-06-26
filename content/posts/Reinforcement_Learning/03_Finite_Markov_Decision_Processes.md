---
title: "Finite Markov Decision Processes"
date: "2022-05-24"
tags: ["Finite Markov Decision Processes"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---

MDPs are a mathematically idealized form of the reinforcement learning problem for which precise theoretical statements can be made. We introduce key elements of the problem's mathematical structure, such as returns, value functions, and Bellman equations. We try to convey the wide range of applications that can be formulated as finite MDPs. As in all of artificial intelligence, there is a tension between breadth of applicability and mathematical tractability.

### The Agent–Environment Interface

MDPs are meant to be a straightforward framing of the problem of learning from interaction to achieve a goal. The learner and decision maker is called the ***agent***. The thing it interacts with, comprising everything outside the agent, is called the ***environment***. These interact continually, the agent selecting actions and the environment responding to these actions and presenting new situations to the agent.1 The environment also gives rise to rewards, special numerical values that the agent seeks to maximize over time through its choice of actions.

<div align="center">
  <img src="/img_RL/03_Agent_envir.PNG" width=650px/>
</div>
<br>

More specifically, the agent and environment interact at each of a sequence of discrete time steps, `$t=0,1,2,3,...$` At each time step `$t$`, the agent receives some representation of the environment’s state, `$S_{t}\in \mathcal{S}$`, and on that basis selects an action, At `$A_{t}\in\mathcal{A}(s)$`. One time step later, in part as a consequence of its action, the agent receives a numerical reward, `$R_{t+1}\in \mathbb{R}$`, and finds itself in a new state, `$S_{t+1}$`. The MDP and agent together thereby give rise to a sequence or trajectory that begins like this:

`$$
S_{0},A_{0},R_{1},S_{1},A_{1},R_{2},S_{2},A_{2},R_{3},... \\
$$`

In a finite MDP, the sets of states, actions, and rewards (`$\mathcal{S}$`, `$\mathcal{A}$`, and `$\mathcal{R}$`) all have a finite number of elements. In this case, the random variables `$R_{t}$` and `$S_{t}$` have well defined discrete probability distributions dependent only on the preceding state and action. That is, for particular values of these random variables, `$s'\in \mathcal{S}$` and `$r\in\mathcal{R}$`, there is a probability of those values occurring at time `$t$`, given particular values of the preceding state and action:

`$$
p(s',r|s,a)\doteq \mathrm{Pr}({S_{t}=s',R_{t}=r}|S_{t-1}=s,A_{t-1}=a) \\
$$`

for all `$s',s\in\mathcal{S},r\in\mathcal{R},$` and `$a \in \mathcal{A}(s)$`. The function `$p$` defines the dynamics of the MDP. The dot over the equals sign in the equation reminds us that it is a definition (in this case of the function `$p$`) rather than a fact that follows from previous definitions. The dynamics function `$p:\mathcal{S}\times\mathcal{R}\times\mathcal{S}\times\mathcal{R} \to [0,1]$` is an ordinary deterministic function of four arguments. The `$|$` in the middle of it comes from the notation for conditional probability, but here it just reminds us that `$p$` specifies a probability distribution for each choice of `$s$` and `$a$`, that is, that

`$$
\sum_{s'\in\mathcal{S}}\sum_{r\in\mathcal{R}}p(s',r|s,a)=1, \mathrm{for} \ \mathrm{all} \ s \in \mathcal{S}, a \in \mathcal{A}(s) \\
$$`

In a ***Markov decision process***, the probabilities given by `$p$` completely characterize the environment’s dynamics. That is, the probability of each possible value for `$S_{t}$` and `$R_{t}$` depends on the immediately preceding state and action, `$S_{t-1}$` and `$A_{t-1}$`, and, given them, not at all on earlier states and actions. This is best viewed as a restriction not on the decision process, but on the ***state***. The state must include information about all aspects of the past agent–environment interaction that make a difference for the future. If it does, then the state is said to have the ***Markov property***.

From the four-argument dynamics function, `$p$`, one can compute anything else one might want to know about the environment, such as the ***state-transition probabilities*** (which we denote, with a slight abuse of notation, as a three-argument function `$p:\mathcal{S}\times\mathcal{S}\times\mathcal{A} \to [0,1]$`),

`$$
p(s'|s,a)\doteq \mathrm{Pr}\{S_{t}=s'|S_{t-1}=s,A_{t-1}=a\}=\sum_{r\in\mathcal{R}}p(s',r|s,a) \\
$$`

We can also compute the expected rewards for state–action pairs as a two-argument function `$r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$`:

`$$
r(s,a)\doteq \mathbb{E}[R_{t}|S_{t-1}=s,A_{t-1}=a]=\sum_{r\in\mathcal{R}}r\sum_{s'\in\mathcal{S}}p(s',r|s,a) \\
$$`

and the expected rewards for state–action–next-state triples as a three-argument function `$r:\mathcal{S}\times\mathcal{A}\times\mathcal{S} \to \mathbb{R}$`,

`$$
r(s,a,s')\doteq \mathbb{E}[R_{t}|S_{t-1}=s,A_{t-1}=a,S_{t}=s]=\sum_{r\in\mathcal{R}}r\frac{p(s',r|s,a)}{p(s'|s,a)} \\
$$`

In particular, the boundary between agent and environment is typically not the same as the physical boundary of a robot’s or an animal’s body. Usually, the boundary is drawn closer to the agent than that. The general rule we follow is that anything that *cannot* be changed arbitrarily by the agent is considered to be **outside of it and thus part of its environment**. We do not assume that everything in the environment is unknown to the agent.

The MDP framework is a considerable abstraction of the problem of goal-directed learning from interaction. It proposes that whatever the details of the sensory, memory, and control apparatus, and whatever objective one is trying to achieve, any problem of learning goal-directed behavior can be reduced to **three signals passing back and forth between an agent and its environment**: one signal to represent the choices made by the **agent** (the actions), one signal to represent the basis on which the choices are made (the states), and one **signal** to define the agent’s goal (the rewards). This framework may not be sufficient to represent all decision-learning problems usefully, but it has proved to be widely useful and applicable.

### Goals and Rewards

In reinforcement learning, the purpose or goal of the agent is formalized in terms of a special signal, called the ***reward***, passing from the environment to the agent. At each time step, the reward is a simple number, `$R_{t} \in \mathbb{R}$`. Informally, the agent’s goal is to maximize the total amount of reward it receives. **This means maximizing not immediate reward, but cumulative reward in the long run.** We can clearly state this informal idea as the ***reward hypothesis***:

> *That all of what we mean by goals and purposes can be well thought of as the maximization of the expected value of the cumulative sum of a received scalar signal (called reward).*

The use of a reward signal to formalize the idea of a goal is one of the most distinctive features of reinforcement learning.

Although formulating goals in terms of reward signals might at first appear limiting, in practice it has proved to be flexible and widely applicable. We can see the agent always learns to maximize its reward. If we want it to do something for us, we must provide rewards to it in such a way that in maximizing them the agent will also achieve our goals. It is thus critical that the rewards we set up truly indicate what we want accomplished. In particular, the reward signal is not the place to impart to the agent prior knowledge about how to achieve what we want it to do.

### Returns and Episodes

We have said that the agent’s goal is to **maximize the cumulative reward** it receives in the long run. How might this be defined formally? If the sequence of rewards received after time step `$t$` is denoted `$R_{t+1},R_{t+2},R_{t+3},...,$` then what precise aspect of this sequence do we wish to maximize? In general, we seek to maximize the ***expected return***, where the return, denoted `$G_{t}$`, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

`$$
G_{t}\doteq R_{t+1}+R_{t+2}+R_{t+3}+...+R_{T}, \\
$$`

where `$T$` is a final time step. This approach makes sense in applications in which there is a natural notion of final time step, that is, when the agent–environment interaction breaks naturally into subsequences, which we call ***episodes***. Each episode ends in a special state called the ***terminal state***, followed by a reset to a standard starting state or to a sample from a standard distribution of starting states. Even if you think of episodes as ending in different ways, such as winning and losing a game, the next episode begins independently of how the previous one ended. Thus the episodes can all be considered to end in the same terminal state, with different rewards for the different outcomes. Tasks with episodes of this kind are called ***episodic tasks***. In episodic tasks we sometimes need to distinguish the set of all nonterminal states, denoted `$\mathcal{S}$`, from the set of all states plus the terminal state, denoted `$\mathcal{S}^{+}$`. The time of termination, `$T$`, is a random variable that normally varies from episode to episode. On the other hand, in many cases the agent-environment interaction does not break naturally into identifiable episodes, but goes on continually without limit.

The additional concept that we need is that of ***discounting***. According to this approach, the agent tries to select actions so that the sum of the discounted rewards it receives over the future is maximized. In particular, it chooses `$A_{t}$` to maximize the expected ***discounted return***:

`$$
G_{t}\doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+...=\sum^{\infty}_{k=0}\gamma^{k}R_{t+k+1}, \\
$$`

where `$\gamma$` is a parameter, `$0\leq\gamma\leq1$`, called the ***discount rate***.

The discount rate determines the present value of future rewards: a reward received `$k$` time steps in the future is worth only `$\gamma^{k-1}$` times what it would be worth if it were received immediately. If `$\gamma <1$`, the infinite sum in the equation has a finite value as long as the reward sequence `$\{R_{k}\}$` is bounded. If `$\gamma=0$`, the agent is “**myopic**” in being concerned only with maximizing immediate rewards: its objective in this case is to learn how to choose `$A_{t}$` so as to maximize only `$R_{t+1}$`. But in general, acting to maximize immediate reward can reduce access to future rewards so that the return is reduced. As `$\gamma$` approaches 1, the return objective takes future rewards into account more strongly; the agent becomes more **farsighted**.

Returns at successive time steps are related to each other in a way that is important for the theory and algorithms of reinforcement learning:

`$$
\begin{align}
G_{t}&\doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2}R_{t+3}+\gamma^{3}R_{t+4}+...\\
&=R_{t+1}+\gamma (R_{t+2}+\gamma R_{t+3}+\gamma^{2}R_{t+4}+...)\\
&=R_{t+1}+\gamma G_{t+1} \\
\end{align}
$$`

Note that this works for all time steps `$t<T$` , even if termination occurs at `$t+1$`, provided we define `$G_{T}=0$`. This often makes it easy to compute returns from reward sequences.

Note that although the return of the equation is a sum of an infinite number of terms, it is still finite if the reward is nonzero and constant — if `$\gamma<1$`. For example, if the reward is a constant +1, then the return is

`$$
G_{t}=\sum^{\infty}_{k=0}\gamma^{k}=\frac{1}{1-\gamma} \\
$$`

### Policies and Value Functions

Almost all reinforcement learning algorithms involve estimating ***value functions***—functions of states (or of state–action pairs) that estimate *how good* it is for the agent to be in a given state (or how good it is to perform a given action in a given state). The notion of “how good” here is defined in terms of future rewards that can be expected, or, to be precise, in terms of expected return. Of course the rewards the agent can expect to receive in the future depend on what actions it will take. Accordingly, value functions are defined with respect to particular ways of acting, called ***policies***.

Formally, a policy is a mapping from states to probabilities of selecting each possible action. If the agent is following policy `$\pi$` at time `$t$`, then `$\pi(a|s)$` is the probability that `$A_{t}=a$` if `$S_{t}=s$`. Like `$p$`, `$\pi$` is an ordinary function; the `$|$` in the middle of `$\pi(a|s)$` merely reminds us that it defines a probability distribution over `$a\in\mathcal{A}(s)$` for each `$s\in\mathcal{S}$`. Reinforcement learning methods specify how the agent’s policy is changed as a result of its experience.

The value function of a state s under a policy `$\pi$`, denoted `$v_{\pi}(s)$`, is the expected return when starting in `$s$` and following `$\pi$` thereafter. For MDPs, we can define `$v_{\pi}$` formally by

`$$
v_{\pi}(s)\doteq \mathbb{E}_{\pi}[G_{t}|S_{t}=s]=\mathbb{E}_{\pi}[\sum^{\infty}_{k=0}\gamma^{k}R_{t+k+1}|S_{t}=s], \mathrm{for} \ \mathrm{all} \ s \in \mathcal{S}, \\
$$`

where `$\mathbb{E}_{\pi}[\cdot]$` denotes the expected value of a random variable given that the agent follows policy `$\pi$`, and `$t$` is any time step. Note that the value of the terminal state, if any, is always zero. We call the function `$v_{\pi}$` the *state-value function for policy* `$\pi$`.

Similarly, we define the value of taking action `$a$` in state `$s$` under a policy `$\pi$`, denoted `$q_{\pi}(s,a)$`, as the expected return starting from `$s$`, taking the action `$a$`, and thereafter following policy `$\pi$`:

`$$
q_{\pi}(s,a)\doteq \mathbb{E}_{\pi}[G_{t}|S_{t}=s,A_{t}=a]=\mathbb{E}_{\pi}[\sum^{\infty}_{k=0}\gamma^{k}R_{t+k+1}|S_{t}=s,A_{t}=a], \\
$$`

We call `$q_{\pi}(s,a)$` the action-value function for policy ``$\pi$``.

A fundamental property of value functions used throughout reinforcement learning and dynamic programming is that they satisfy recursive relationships similar to that which we have already established for the return. For any policy ``$\pi$`` and any state ``$s$``, the following consistency condition holds between the value of ``$s$`` and the value of its possible successor states:

`$$
\begin{align}
v_{\pi}(s)&\doteq\mathbb{E}[G_{t}|S_{t}=s] \\
&=\mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_{t}=s] \\
&= \sum_{a}\pi(a|s)\sum_{s'}\sum_{r}p(s',r|s,a)[r+\gamma\mathbb{E}_{\pi}[G_{t+1}|S_{t+1}=s']] \\
&= \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r+\gamma v_{\pi}(s')], \mathrm{\ for \ all \ }s\in\mathcal{S} \\
\end{align}
$$`

where it is implicit that the actions, ``$a$``, are taken from the set ``$\mathcal{A}(s)$``, that the next states, ``$s'$``, are taken from the set ``$\mathcal{S}$``, and that the rewards, ``$r$``, are taken from the set ``$\mathcal{R}$``. Note also how in the last equation we have merged the two sums, one over all the values of ``$s'$`` and the other over all the values of ``$r$``, into one sum over all the possible values of both. We use this kind of merged sum often to simplify formulas. Note how the final expression can be read easily as an expected value. It is really a sum over all values of the three variables, ``$a$``, ``$s'$``, and ``$r$``. For each triple, we compute its probability, ``$\pi(a,s)p(s',r|s,a)$``, weight the quantity in brackets by that probability, then sum over all possibilities to get an expected value.

Previous equation is the ***Bellman equation*** for ``$v_{\pi}$``. It expresses a relationship between the value of a state and the values of its successor states. Think of looking ahead from a state to its possible successor states, as suggested by the diagram to the right. Each open circle represents a state and each solid circle represents a state–action pair. Starting from state $s$, the root node at the top, the agent could take any of some set of actions — three are shown in the diagram — based on its policy ``$\pi$``. From each of these, the environment could respond with one of several next states, ``$s'$`` (two are shown in the figure), along with a reward, ``$r$``, depending on its dynamics given by the function $p$. The Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way. The value function ``$v_{\pi}$`` is the **unique solution to its Bellman equation**.

<div align="center">
  <img src="/img_RL/03_Backup.PNG" width=250px/>
</div>
<br>

### Optimal Policies and Optimal Value Functions

Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of reward over the long run. For finite MDPs, we can precisely define an optimal policy in the following way. Value functions define a partial ordering over policies. A policy ``$\pi$`` is defined to be better than or equal to a policy ``$\pi'$`` if its expected return is greater than or equal to that of ``$\pi'$`` for all states. In other words, ``$\pi\geq\pi'$`` if and only if ``$v_{\pi}(s)\geq v_{\pi'}(s)$`` for all ``$s\in\mathcal{S}$``. There is always at least one policy that is better than or equal to all other policies. This is an ***optimal policy***. Although there may be more than one, we denote all the optimal policies by ``$\pi_{*}$``. They share the same state-value function, called the ***optimal state-value function***, denoted ``$v_{*}$``, and defined as

`$$
v_{*}(s)\doteq\max_{\pi}v_{\pi}(s), \mathrm{\ for \ all \ } s\in\mathcal{S} \\
$$`

Optimal policies also share the same ***optimal action-value function***, denoted ``$q_{*}$``, and defined as

`$$
q_{*}(s,a)\doteq\max_{\pi}q_{\pi}(s,a), \mathrm{\ for \ all \ } s\in\mathcal{S}\mathrm{\ and\ } a \in \mathcal{A}  \\
$$`

For the state–action pair ``$(s,a)$``, this function gives the expected return for taking action ``$a$`` in state ``$s$`` and thereafter following an optimal policy. Thus, we can write ``$q_{*}$`` in terms of ``$v_{*}$`` as follows:

`$$
q_{*}(s,a)=\mathbb{E}[R_{t+1}+\gamma v_{*}(S_{t+1})|S_{t}=s, A_{t}=a] \\
$$`

### Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
