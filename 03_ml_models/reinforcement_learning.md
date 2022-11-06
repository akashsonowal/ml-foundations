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

Policy can be either reaching the shortest terminal state fast or left or right or go to nearest.

$state -(policy) \pi -> action taken $

Policy is to pick the actions that maxmises the return.

$\pi(s) = a$

## Quiz 1


# State-Action Value Function

## Quiz 2

# Continuous State Space

## Quiz 3

# LAB
