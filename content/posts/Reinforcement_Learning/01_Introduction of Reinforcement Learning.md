---
title: "Introduction of Reinforcement Learning"
date: "2022-05-14"
tags: ["Introduction of Reinforcement Learning"]
categories: ["Data Science", "Reinforcement Learning"]
weight: 3
---


> **Reinforcement learning** is learning what to do—how to map situations to actions—so as to maximize a numerical reward signal. The learner is not told which actions to take, but instead must discover which actions yield the **most reward(最大奖励回报)** by trying them. In the most interesting and challenging cases, actions may affect not only the immediate reward but also the next situation and, through that, all subsequent rewards. These two characteristics—trial-and-error search and delayed reward—are the two most important distinguishing features of reinforcement learning.

**Reinforcement Learning (RL)** is an Artificial Intelligence (AI) discipline—it is concerned with the engineering of mechanisms to enable an agent to competently act autonomously in an environment of which the agent has imperfect knowledge and imperfect control over.

**Competence** implies performance; the **agent** is attempting to solve a problem in the environment with some quantitative, objective standard of performance. That is, it is attempting to optimally achieve a result in this environment, where the notion of optimality demands quantitative metrics, and a demonstration that those metrics are achieving some maximum due to the agent’s actions.

**Autonomy** implies that the agent selects its own actions without remote control by another agent or human. The agent bases these actions on sensor feedback from the environment. The agent is thus solving a control problem. In contrast to classical control theoretic methods that require knowledge of the environment’s dynamical equations, reinforcement learning methods are geared towards situations where this knowledge is **absent**, **minimal**, or **intractable**. The leads us to the design of agents that:

- **exploit** their instantaneous knowledge of the **environment**
- refine their knowledge of the environment by experience, which includes **exploration** of the environment

to solve the action selection problem.

## Reinforcement Learning and Machine Learning

<div align="center">
  <img src="/img_RL/01_Overview.PNG" width=650px/>
</div>
<br>

Reinforcement learning is different from ***supervised learning***, the kind of learning studied in most current research in the field of machine learning. Supervised learning is learning from a **training set of labeled examples** provided by a knowledgable external supervisor. Each example is a description of a situation together with a specification—the label—of the correct action the system should take in that situation, which is often to identify a category to which the situation belongs. The object of this kind of learning is for the system to extrapolate, or generalize, its responses so that it acts correctly in situations not present in the training set. This is an important kind of learning, but alone it is ***not* adequate for learning from interaction**. In interactive problems it is often impractical to obtain examples of desired behavior that are both correct and representative of all the situations in which the agent has to act. In uncharted territory—where one would expect learning to be most beneficial—an agent must be able to learn from its own experience.

Reinforcement learning is also different from what machine learning researchers call ***unsupervised learning***, which is typically about finding structure hidden **in collections of unlabeled data**. The terms supervised learning and unsupervised learning would seem to exhaustively classify machine learning paradigms, but they do not. Although one might be tempted to think of reinforcement learning as a kind of unsupervised learning because it does not rely on examples of correct behavior, reinforcement learning is trying to **maximize a reward signal** instead of trying to find hidden structure. Uncovering structure in an agent’s experience can certainly be useful in reinforcement learning, but by itself does not address the reinforcement learning problem of maximizing a reward signal. We therefore consider reinforcement learning to be a third machine learning paradigm, alongside supervised learning and unsupervised learning and perhaps other paradigms.

## Challenges

One of the challenges that arise in reinforcement learning, and not in other kinds of learning, is the trade-off between **exploration(探索)** and **exploitation(利用)**. To obtain a lot of reward, a reinforcement learning agent must prefer actions that it has tried in the past and found to be effective in producing reward. But to discover such actions, it has to try actions that it has not selected before. The agent has to exploit what it has already experienced in order to obtain reward, but it also has to explore in order to make better action selections in the future. The dilemma is that neither exploration nor exploitation can be pursued exclusively without failing at the task. The agent must try a variety of actions and progressively favor those that appear to be best. On a stochastic task, each action must be tried many times to gain a reliable estimate of its expected reward.

Another key feature of reinforcement learning is that it explicitly considers the whole problem of a goal-directed agent interacting with an uncertain environment. This is in contrast to many approaches that consider subproblems without addressing how they might fit into a larger picture.

Reinforcement learning takes the opposite tack, starting with a **complete**, **interactive**, **goal-seeking** agent. All reinforcement learning agents have explicit goals, can sense aspects of their environments, and can choose actions to influence their environments. Moreover, it is usually assumed from the beginning that the agent has to operate despite significant uncertainty about the environment it faces. When reinforcement learning involves planning, it has to address the interplay between planning and real-time action selection, as well as the question of how environment models are acquired and improved. When reinforcement learning involves supervised learning, it does so for specific reasons that determine which capabilities are critical and which are not. For learning research to make progress, important subproblems have to be isolated and studied, but they should be subproblems that play clear roles in complete, interactive, goal-seeking agents, even if all the details of the complete agent cannot yet be filled in.

## Elements of Reinforcement Learning

Beyond the agent and the environment, one can identify four main subelements of a reinforcement learning system: a ***policy***, a ***reward signal***, a ***value function***, and, optionally, a ***model*** of the environment.

A ***policy*** defines the learning agent’s way of behaving at a given time. Roughly speaking, a policy is a mapping from perceived states of the environment to actions to be taken when in those states. In some cases the policy may be a simple function or lookup table, whereas in others it may involve extensive computation such as a search process. The policy is the **core** of a reinforcement learning agent in the sense that it alone is sufficient to determine behavior. In general, policies may be **stochastic**, specifying probabilities for each action.

A ***reward signal*** defines the goal of a reinforcement learning problem. On each time step, the environment sends to the reinforcement learning agent a single number called the **reward**. The agent’s sole objective is to maximize the total reward it receives over the long run. The reward signal thus defines what are the good and bad events for the agent. The reward signal is the primary basis for altering the policy; if an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future. In general, reward signals may be stochastic functions of the state of the environment and the actions taken.

Whereas the reward signal indicates what is good in an immediate sense, a ***value function*** specifies what is good in the long run. Roughly speaking, the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Whereas rewards determine the immediate, intrinsic desirability of environmental states, values indicate the long-term desirability of states after taking into account the states that are likely to follow and the rewards available in those states.

Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Nevertheless, it is **value**s with which we are **most concerned** when making and evaluating decisions. Action choices are made based on value judgments. We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run. Unfortunately, it is much harder to determine values than it is to determine rewards.

The fourth and final element of some reinforcement learning systems is a ***model*** of the environment. This is something that mimics the behavior of the environment, or more generally, that allows inferences to be made about how the environment will behave.

## Examples

- A master chess player makes a move. The choice is informed both by planning— anticipating possible replies and counterreplies—and by immediate, intuitive judg- ments of the desirability of particular positions and moves.
- A gazelle calf struggles to its feet minutes after being born. Half an hour later it is running at 20 miles per hour.
- A mobile robot decides whether it should enter a new room in search of more trash to collect or start trying to find its way back to its battery recharging station. It makes its decision based on the current charge level of its battery and how quickly and easily it has been able to find the recharger in the past.

These examples share features that are so basic that they are easy to overlook. All involve **interaction** between an active **decision-making agent** and its **environment**, within which the agent seeks to achieve a *goal* despite *uncertainty* about its environment.

At the same time, in all of these examples the effects of actions *cannot* be fully predicted; thus the agent must monitor its environment frequently and react appropriately. Agent can also use its **experience** to improve its performance over time.


## Summary

Reinforcement learning is a computational approach to understanding and automating goal-directed learning and decision making. It is distinguished from other computational approaches by its emphasis on learning by an agent from direct interaction with its environment, without requiring exemplary supervision or complete models of the environment. In our opinion, reinforcement learning is the first field to seriously address the computational issues that arise when learning from interaction with an environment in order to achieve long-term goals.



## Reference

[1] Sutton, R. S., Bach, F., &amp; Barto, A. G. (2018). Reinforcement learning: An introduction. MIT Press Ltd.
