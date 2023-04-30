- how to predict the rating a user might give to a prospective item
- how to generate a recommendation list of items0
- how to predict the click-through rate from abundant features.

Overview of Recommender Systems
- Collaborative Filtering: 
  the data is user-item data.
  - memory based (nearest neighbors based CF: user based or item based). It has limitation of dealing with sparse and large dataset.
  - model based (latent factor models: matrix factorization). It deals with sparse and large dataset effectively. Many of these even can be extended with NN. 
  - and their hybrid
- Content based systems: contents descriptions of items/users
- Context based systems: context like timestamps and locations
- Explicit Feedback and Implicit Feedback:
  explicit: youtube likes, imdb ratings. It is hard to collect as user needs to proactive to give the feedbacks.
  implicit: user clicks, purchase history, browsing history, watches and even mouse movements. The implicit data is very noisy.
 - Recommendation tasks
   - movies recommendation, news recommendation, point-of-interest recommendation. 
   - based on tasks of feedback and input data. 
   - rating prediction task predicts explicit ratings.
   - Top-n recommendation (item ranking) ranks all items for each user personally based on implicit feedback.
   - If time stamp info is also incuded, we can build sequence aware recommendation.
   - click through rate prediction (implicit feedbackwith many categorical features)
   - recommendation for new users and new items to existing users (cold start recommendation).
 
## Movie Lena Dataset (100k) 
| user_id | item_id | rating | timestamp|
this is the ratings dataset

after this construct the interative matrix of size n x m. we can calculate the sparsity of this matrix using 1- (non-zero entries) / (num_users * num_items).

we use additional side info such as user/item features to alleviate the sparsity.

plot the distribution of count of different ratings.

split the ratings dataset (random and sequence aware: newest for test and hostoricl for train)

make users, items, scores and interactions.

## Matrix Factorization
factoring the interaction matrix

 
