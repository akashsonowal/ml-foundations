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
## Quiz 1

Question 1
You are using reinforcement learning to control a four legged robot. The position of the robot would be its _state____.

Question 2
You are controlling a Mars rover. You will be very very happy if it gets to state 1 (significant scientific discovery), slightly happy if it gets to state 2 (small scientific discovery), and unhappy if it gets to state 3 (rover is permanently damaged). To reflect this, choose a reward function so that:

R(1) > R(2) > R(3) where R(1) and R(2) is positive and R(3) Is negative

Question 3
You are using reinforcement learning to fly a helicopter. Using a discount factor of 0.75, your helicopter starts in some state and receives rewards -100 on the first step, -100 on the second step, and 1000 on the third and final step (where it has reached a terminal state). What is the return?
-0.75*100 - 0.75^2*100 + 0.75^3*1000 

Question 4
Given the rewards and actions below, compute the return from state 3 with a discount factor of $\gamma = 0.25$.

6.25





# State-Action Value Function

## Quiz 2

# Continuous State Space

## Quiz 3

# LAB
