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

obj_pos_1, obj_pos_2, obj_pos_3 = p1

print(f"object at position 1 is '{obj_pos_1}'")
print(f"object at position 2 is '{obj_pos_2}'")
print(f"object at position 3 is '{obj_pos_3}'")

score_obj_pos_1 = scores_dict[obj_pos_1]
score_obj_pos_2 = scores_dict[obj_pos_2]
score_obj_pos_3 = scores_dict[obj_pos_3]

first_term_numerator = np.exp(score_obj_pos_1)
first_term_denominator = np.exp(score_obj_pos_1) + np.exp(score_obj_pos_2) + np.exp(score_obj_pos_3)

first_term = first_term_numerator / first_term_denominator

print(f"first term is {first_term}")

second_term_numerator = np.exp(score_obj_pos_2)
second_term_denominator = np.exp(score_obj_pos_2) + np.exp(score_obj_pos_3)

second_term = second_term_numerator / second_term_denominator

print(f"second term is {second_term}")

third_term = 1.0

prob_of_permutation = first_term * second_term * third_term

print(f"probability of permutation is {prob_of_permutation}")

top_1_shirt_prob = np.exp(scores_dict['shirt']) / sum(np.exp(list(scores_dict.values())))

ordered_scores = np.array([scores_dict[x] for x in xlabs]).astype(np.float32)
predicted_prob_dist = tf.nn.softmax(ordered_scores)

print(predicted_prob_dist)

raw_relevance_grades = tf.constant([3.0, 1.0, 0.0], dtype=tf.float32)
true_prob_dist = tf.nn.softmax(raw_relevance_grades)

print(true_prob_dist)

kl_div = sum(true_prob_dist * np.log(true_prob_dist / predicted_prob_dist))

query_1 = "dog"

bing_search_results = [
    "Dog - Wikipedia",
    "Adopting a dog or puppy | RSPCA Australia",
    "dog | History, Domestication, Physical Traits, & Breeds",
    "New South Wales | Dogs & Puppies | Gumtree Australia Free",
    "dog - Wiktionary"
]

query_2 = "what is a dog"

google_search_results = [
    "Dog - Wikipedia",
    "Dog - Simple English Wikipedia, the free encyclopedia",
    "Dog | National Geographic",
    "dog | History, Domestication, Physical Traits, & Breeds",
    "What is a Dog | Facts About Dogs | DK Find Out"
]

relevance_grades = tf.constant([
    [3.0, 2.0, 2.0, 2.0, 1.0],
    [3.0, 3.0, 1.0, 1.0, 0.0]
])





class ListNet:
    def __init__(self, )

