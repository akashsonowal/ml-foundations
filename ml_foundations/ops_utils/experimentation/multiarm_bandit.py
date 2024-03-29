# https://www.analyticsvidhya.com/blog/2018/09/reinforcement-multi-armed-bandit-scratch-python/

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import random 

def thompson_sampling(dataset):
    N, d = dataset.shape()
    ads_selected = []
    numbers_of_rewards_1 = [0] * d 
    numbers_of_rewards_0 = [0] * d 
    total_reward = 0

    for n in range(N):
        ad = 0
        max_random = 0

        for i in range(d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)

            if random_beta > max_random:
                max_random = random_beta
                ad = i

        ads_selected,append(ad)
        reward = dataset.values[n, ad]

        if reward == 1:
            numbers_of_rewards_1[ad] += 1
        else:
            numbers_of_rewards_0[ad] += 1
        
        total_reward += reward 

    plt.hist(ads_selected)
    plt.title("Histogram of ads selection")
    plt.xlabel("Ads")
    plt.ylabel("Number of times each ad was selected")
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
    thompson_sampling(dataset)