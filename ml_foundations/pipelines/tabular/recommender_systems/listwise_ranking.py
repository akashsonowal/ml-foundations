# listnet ranking adopted from https://embracingtherandom.com/machine-learning/tensorflow/ranking/deep-learning/learning-to-rank-part-2/

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

combined_texts = [query_1, *bing_search_results, query_2, *google_search_results]

tokeniser = tf.keras.preprocessing.text.Tokenizer()
tokeniser.fit_on_texts(combined_texts)

# we add one here to account for the padding word
vocab_size = max(tokeniser.index_word) + 1
print(vocab_size)

for idx, word in tokeniser.index_word.items():
    print(f"index {idx} - {word}")

EMBEDDING_DIMS = 2

embeddings = np.random.randn(vocab_size, EMBEDDING_DIMS).astype(np.float32)

print(embeddings)

query_1_embedding_index = tokeniser.texts_to_sequences([query_1])
query_1_embeddings = np.array([embeddings[x] for x in query_1_embedding_index])

print(query_1_embeddings)

query_2_embedding_indices = tokeniser.texts_to_sequences([query_2])
query_2_embeddings = np.array([embeddings[x] for x in query_2_embedding_indices])

print(query_2_embeddings)

query_2_embeddings_avg = tf.reduce_mean(query_2_embeddings, axis=1, keepdims=True).numpy()

print(query_2_embeddings_avg)

query_embeddings = np.row_stack([query_1_embeddings, query_2_embeddings_avg]) # (2, 1, 2)

docs_sequences = []
for docs_list in [bing_search_results, google_search_results]:
    docs_sequences.append(tokeniser.texts_to_sequences(docs_list))

docs_embeddings = []
for docs_set in docs_sequences:
    this_docs_set = []
    for doc in docs_set:
        this_doc_embeddings = np.array([embeddings[idx] for idx in doc])
        this_docs_set.append(this_doc_embeddings)
    docs_embeddings.append(this_docs_set)

for embeddings in docs_embeddings[0]:
    print()
    print(embeddings)

for embeddings in docs_embeddings[1]:
    print()
    print(embeddings)

docs_averaged_embeddings = []
for docs_set in docs_embeddings:
    this_docs_set = []
    for doc in docs_set:
        this_docs_set.append(tf.reduce_mean(doc, axis=0, keepdims=True))
    concatenated_docs_set = tf.concat(this_docs_set, axis=0).numpy()
    docs_averaged_embeddings.append(concatenated_docs_set)
    
docs_averaged_embeddings = np.array(docs_averaged_embeddings)

print(docs_averaged_embeddings.shape) # (2, 5, 2) # 5 docs per query

NUM_DOCS_PER_QUERY = 5

expanded_queries = tf.gather(query_embeddings, [0 for x in range(NUM_DOCS_PER_QUERY)], axis=1).numpy()

print(expanded_queries)

expanded_batch = np.concatenate([expanded_queries, docs_averaged_embeddings], axis=-1)

print(expanded_batch) # (2, 5, 4) # (2 + 2)

dense_1 = tf.keras.layers.Dense(units=3, activation='relu')
dense_1_out = dense_1(expanded_batch)

print(dense_1_out) # (2, 5, 3)

scores = tf.keras.layers.Dense(units=1, activation='linear')
scores_out = scores(dense_1_out)

print(scores_out) # (2, 5, 1)

scores_for_softmax = tf.squeeze(scores_out, axis=-1)
scores_prob_dist = tf.nn.softmax(scores_for_softmax, axis=-1)

print(scores_prob_dist)

relevance_grades_prob_dist = tf.nn.softmax(relevance_grades, axis=-1)

print(relevance_grades_prob_dist)

loss = tf.keras.losses.KLDivergence()
batch_loss = loss(relevance_grades_prob_dist, scores_prob_dist)

print(batch_loss)

per_example_loss = tf.reduce_sum(
    relevance_grades_prob_dist * tf.math.log(relevance_grades_prob_dist / scores_prob_dist),
    axis=-1
)

print(per_example_loss)

batch_loss = tf.reduce_mean(per_example_loss)

print(batch_loss)

# pipeline
NUM_DOCS_PER_QUERY = 5
EMBEDDING_DIMS = 2

class ExpandBatchLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ExpandBatchLayer, self).__init__(**kwargs)
        
    def call(self, input):
        queries, docs = input
        batch, num_docs, embedding_dims = tf.unstack(tf.shape(docs))
        expanded_queries = tf.gather(queries, tf.zeros([num_docs], tf.int32), axis=1)
        return tf.concat([expanded_queries, docs], axis=-1)

query_input = tf.keras.layers.Input(shape=(1, EMBEDDING_DIMS, ), dtype=tf.float32, name='query')
docs_input = tf.keras.layers.Input(shape=(NUM_DOCS_PER_QUERY, EMBEDDING_DIMS, ), dtype=tf.float32, 
                name='docs')

expand_batch = ExpandBatchLayer(name='expand_batch')
dense_1 = tf.keras.layers.Dense(units=3, activation='linear', name='dense_1')
dense_out = tf.keras.layers.Dense(units=1, activation='linear', name='scores')
scores_prob_dist = tf.keras.layers.Dense(units=NUM_DOCS_PER_QUERY, activation='softmax', 
                      name='scores_prob_dist')

expanded_batch = expand_batch([query_input, docs_input])
dense_1_out = dense_1(expanded_batch)
scores = tf.keras.layers.Flatten()(dense_out(dense_1_out))
model_out = scores_prob_dist(scores)

model = tf.keras.models.Model(inputs=[query_input, docs_input], outputs=[model_out])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0.9), 
              loss=tf.keras.losses.KLDivergence())

hist = model.fit(
    [query_embeddings, docs_averaged_embeddings], 
    relevance_grades_prob_dist, 
    epochs=50, 
    verbose=False
)

class ListNet:
    def __init__(self, )
