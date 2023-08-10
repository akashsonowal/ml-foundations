# Upper confidence bound

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

import math

def ucb(dataset):
    N, d = dataset.shape()
    ads_selected = []
    numbers_of_selection = [0] * d 
    sums_of_rewards = [0] * d 
    total_reward = 0 

    for n in range(N): # no of users or iterations
        ad = 0
        max_upper_bound = 0
        for i in range(d):
            if numbers_of_selection[i] >  0:
                average_reward = sums_of_rewards[i] / numbers_of_selection[i]
                delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selection[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i 
        ads_selected.append(ad)
        numbers_of_selection[ad] += 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] += reward
        total_reward += reward 
    
    plt.hist(ads_selected)
    plt.title("Histogram of ads selections")
    plt.xlabel("Ads")
    plt.ylabel("Number of times each ad was selected")
    plt.show()

if __name__ == "__main__":
    dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
    ucb(dataset)


