# Introduction

## Supervised Learning

Optimizing machine learning models based on previously observed features and labels. Typically, the goal is to attach the most likely label to some provided features

## Unsupervised Learning

An approach within machine learning that takes in unlabeled examples and produces patterns from the provided data. Typically, the goal is to discover something previously unknown about the unlabeled examples.

## Deep Learning

Optimizing neural networks, often with many hidden layers, to perform unsupervised or supervised learning.

## Recommendation Systems 

Systems with the goal of presenting an item to a user such that the user will most ikely purchase, view, or like the recommended item. Items can take many forms, suich as music, movies, or products. Also called recommender systems.

## Ranking

Optimizing machine learning models to rank candidates, such as music, articles, or products. Typically, the goal Is to order the candidates such that the candidates which are most likely to be interacted with (purchased, viewed, liked, etc.) are above other candidates that aren't as likely to be interacted with.

# Prerequisite

## Features

A set of quantities or properties describing an observation. They can be binary like "day” and "night"; categorical like "morning”, "afternoon’, and "evening”; continuous like 3.141; or ordinal ike "threatened", "endangered", "extinct’, where the categories can be ordered. 

## Labels

Usually paired with a set of features for use in supervised learning. Can be discrete or continuous.

## Examples

Pairs of features and labels.

## Dimensions
Here, the number of features associated with a particular example. §

| Vector @ |
Here, a feature vector, which is a list of features representing a particular example.

| Matrix @
An array of values, usually consisting of multiple rows and columns. §

| Matrix Transpose @)
An operator that flips a matrix over its diagonal.
| Polynomial @
Afunction with more than one variable/coefficient pair.
| perivative @
Indicates how much the output of a function will change with respect to a change in its input.
| Probability €
How likely something Is to occur. This can be independent, such as the roll of the dice, or conditional, such as drawing two cards
subsequently out of a deck without replacement.
| Probability Distribution @)
Afunction that takes in an outcome and outputs the probability of that particular outcome occurring.
| Gaussian Distribution @)
Avery common type of probability distribution which fits many real world observations; also called a normal distribution.
| uniform Distribution @)
A probability distribution In which each outcome s equally likely; for example, rolling a normal six-sided die.

# Supervised Learning

# Naive Bayes

# Unsupervised Learning

## KMeans
| Unsupervised Learning 1
An approach within machine learning that takes in unlabeled examples and produces patterns from the provided data. Typically, the
goal s to discover something previously unknown about the unlabeled examples. §
I Centroid
The location of the center of a cluster in n-dimensions.
I Inertia
The sum of distances between each point and the centroid.
| Local optima @)
AAmaxima or minima which is not the global optima
| Non-convex Function @)
A function which has two or more instances of zero-slope.
| Elbow Method
Amethod of finding the best value for k in k-means. It Involves finding the elbow of the plot of a range of ks and thelr respective
inertias.
I silhouette Method
A method of finding the best value for k in k-means. It takes into account the ratios of the inter and intra clusters distances.
| K-means++
Using a weighted probability distribution as a way to find the Initial centroid locations for the k-means algorithm.
| Agglomerative Clustering
Aclustering algorithm that builds a hierarchy of subclusters that gradually group into a single cluster. Some techniques for measuring
distances between clusters are single-linkage and complete-linkage methods.
Note: For non-separable data points we can use agglomerative clustering.

## 
| Singular Value Decomposition i
Also SVD, a process which decomposes a matrix into rotation and scaling terms. It is a generalization of eigendecomposition. §
| Rank r Approximation i
Using up to, and including, the rth terms in the singular value decomposition to approximate an original matrix. §
| Dimensionality Reduction i
The process of reducing the dimensionality of features. This is typically useful to speed up the training of models and In some cases,
allow for a wider number of machine learning algorithms to be used on the examples. This can be done with SVD (or PCA) and as well, |
certaln types of neural networks such as autoencoders. §
| Eigendecomposition @) |
Applicable only to square matrices, the method of factoring a matrix into its eigenvalues and eigenvectors. An eigenvector is a vector |
which applies a linear transformation to some matrix being factored. The eigenvalues scale the eigenvector values. 3
| Principal Component Analysis 1
Also PCA, is eigendecomposition performed on the covariance matrix of some particular data. The eigenvectors then describe the §
principle components and the elgenvalues indicate the variance described by each principal component. Typically, VD Is usedto |
perform PCA. PCA assumes linear correlations in the data. If that assumption Is not true, then you can use kernel PCA. §
| orthogonal @
Perpendicular s n-dimensions.

# Deep Learning

# Recommender Systems
| Content Filtering
A recommendation technique which takes into account a single user's features and many items' features.
| Collaborative Filtering
A recommendation technique which uses many user's and item's data typically in the form of a user-item matrix.
| User-item Matrix
/A matrix which contains the Interactions of users and items. Items can take the form of products, music, and videos.
| Pearson Correlation @)
A measure of the correlation between two inputs. In the context of recommendation systems. Pearson correlation can be used to
construct an item-item similarity matrix.
| Time Decay
The added modelling assumption that interactions between items and users should count for less over time.
I Inverse User Frequency
The added modelling assumption that if a user interacts with a less overall popular item, then the interaction should count for more.
| Matrix Factorization
Factors the user-item matrix into embeddings such that multiplying together the resulting embedding layers gives an estimate for the
original user-item matrix.
| Implicit Rating 1
A rating obtained from user behavior as opposed to surveying the user. 1
| sparkML 1
Refers to APIs which provide machine learning capabilities on Spark dataframes. 1
| Cold-start 1
A challenge with recommendation systems in which users or items are new and that there Is limited or no information in terms of the |
user-item matrix. This can make personalized recommendations difficult or impossible. §
| Echo Chamber 1
Astate of arecommendation system in which user behavior is reinforced by the recommendations themselves. 3
I shilling Attack 3
Atype of attack on a recommendation system in which users manipulate the recommendations by inflating or deflating positive 3
Interactions for their own or competing items. §

# Ranking
| Candidate Generator 1
A system which outputs the candidates to be ranked. This is sometimes referred to as retrieving the ‘top-k' §
| Embedding Space 1
The n-dimensional space where embeddings exist. Typically, the embedding space can be used to generate the top-k candidates by
using the k-nearest neighbors algorithm. §
| cosine similarity @) |
A similarity metric which is equal to the cosine of the angle between two inputs in some embedding space.
| Linear Activation
A symmetric activation function which assigns the output as the value of the input.
| Normalized Discounted Cumulative Gain
An Information retrieval metric which assigns a value of a particular ranking based on each Item's position and relevance.
| Mean Average Precision
Abinary ranking metric which takes into account the relevance of ranked items with regards to their position. 3
| Mean Reciprocal Rank
Abinary ranking metric which takes Into account the first spot In a ranking which contains a relevant item §
| Shrinkage
Alearning rate for gradient boosted trees. §
| side Features

Features in addition to item and user embeddings. This can include properties of items and users.
| Implicit Relevance Feedback

Feedback obtained from user behavior as opposed to surveying the user.
| Presentation and Trust Bias

Biases found within ranking which arise from the placement of items within a ranking.

- Cover notes
- Code implementation from scratch
