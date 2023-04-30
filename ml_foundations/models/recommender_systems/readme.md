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

split the ratings dataset (random and sequence aware: newest for test and hostoricl for train):
each train data and test data has the same columns as ratings dataset. in temporal, same user data should be present in train and test set (offcourse the latest one in test)

make users, items, scores and interactions. the users and items are list of user_index and item_index from spliting the column in dataframe. if explicit then interactions, then inter is a 2d matrix of user-item interactions. If feedback is implicit then inter is {user_index: iten_index} and scores is list of values of 1.

Now load the users array, items array and scores array into dataloader with batch size....don't touch iteractions.

## Matrix Factorization
factoring the interaction matrix
now we train a matrix factorization model with X as (user, item) and Y as scores.
matrix factor model takes the user items in batch and geting the embedding for it. and same with items...then dot-product and take sum in latent axis + add biases to get a 1d array of scores of that batch.

## AutoRec: Rating Predictions with AutoEncoders

## Personalized Ranking for Recommender Systems

## Neural Collaborative Filtering for Personalized Ranking

## Sequence Aware Recommender Systems

## Feature Rich Recommender Systems

## Factorization Machines

## Deep Factorization Machines

 
