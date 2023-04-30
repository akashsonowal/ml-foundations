- [Free Resource](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera)

# Introduction

state -> action

x -> y

given a state what actions to take? This cannot be supervised becuase we don't know optimal y but we can learn through rewadr function

reward function to teach how to take good actions

control robots
factory optimization
financial trading
playing games

2. 

(S, a, R(S), S')

R(s) is the reward associted at state S.

We traverse till we reach the terminal state.

Return helps us take into account time also to make our algorithm impatinet.

$$ Return = R_{1} + \gamma R_{2} + \gamma^{2} R_{3} + \gamma^{3} R_{4} + ... (until terminal state) $$ 

Return is the return calculated if it starts at start state and $R_{i}$ is the reward associated at state S. 

Rewards depend on actions and returns depend on reward thus reward depend on actions.


So, $\gamma$ defines the less weightage to later states.

The action can be mix of both left and right.

Policy can be either reaching the shortest terminal state fast or left or right or go to nearest. In simple words a policy can be to maximise the returns etc or to learn more function.

$state -(policy) \pi -> action taken $

Policy is to pick the actions that maxmises the return.

Find $\pi(s) = a$

Markov Decision Proces

The above formulaism is known as MDP. It uses the concepts of the reward, discountfactor, return, policy. 
The future only depends on curent state but not in everything that might have occured
## [Quiz 1: RL Introduction](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/eb7aab8b6964336d3d8569f6e9380ca83775969e/C3%20-%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/week3/Practice%20quiz%20:%20Reinforcement%20learning%20introduction)


# State-Action Value Function

Q(s,a) is at state s take action a and behave optimally after that.

for every state we can determine the Q(s, a) value depending on the action taken.

The best action is maxa(Q(s, a)

so $\pi(s) = a$ is the a that maximises the Q(s, a)

the gamma values determines the patience. High gamma means more patience and low gamma means less patience.

Bellman equation formalizes the definition of Q(s, a) value function.

IN real world our robot may not go optimially so we have expected return of the series of actions taken.

[State-Action Value Function: Optional Lab]()

## [Quiz 2](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/eb7aab8b6964336d3d8569f6e9380ca83775969e/C3%20-%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/week3/Practice%20Quiz%20:%20State-action%20value%20function)

# Continuous State Space

The state space can be continuous. e.g. (x, y, theta, velocity)

we train a neural network to compute the Q(s, a)

## [Quiz 3](https://github.com/greyhatguy007/Machine-Learning-Specialization-Coursera/tree/eb7aab8b6964336d3d8569f6e9380ca83775969e/C3%20-%20Unsupervised%20Learning%2C%20Recommenders%2C%20Reinforcement%20Learning/week3/Practice%20Quiz%20:%20Continuous%20state%20spaces)

# LAB
