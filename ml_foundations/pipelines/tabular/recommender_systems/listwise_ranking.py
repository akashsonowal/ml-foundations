import random
random.seed(1)

import numpy as np 
np.random.seed(1)

import itertools 
import matplotlib.pyplot as plt 
plt.style.use("ggplot")

import tensorflow as tf 


objects_to_rank = {'dress', 'shirts', 'pants'}
all_permutations = list(itertools.permutations(objects_to_rank))
for x in sorted(all_permutations):
    print(x)

scores_dict = {x: np.random.randn(1)[0] for x in ['shirt', 'pants', 'dress']}  
print(scores_dict)

p1 = random.choice(all_permutations)
print(p1)

obj_pos_1, obj_pos_2, obj_pos_3 = pi

print(f"object at position 1 is '{obj_pos_1}'")
print(f"object at position 2 is '{obj_pos_2}'")
print(f"object at position 3 is '{obj_pos_3}'")








class ListNet:
    def __init__(self, )

