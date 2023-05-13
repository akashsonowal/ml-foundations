# [Code Implementations from Scratch] Models

## Basic Models

## Ranking
- https://medium.com/data-science-at-microsoft/search-and-ranking-for-information-retrieval-ir-5f9ca52dd056


## Deep Models

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
| Probability Distribution @
Afunction that takes in an outcome and outputs the probability of that particular outcome occurring.
| Gaussian Distribution @
Avery common type of probability distribution which fits many real world observations; also called a normal distribution.
| uniform Distribution @
A probability distribution In which each outcome s equally likely; for example, rolling a normal six-sided die.

# Supervised Learning

# Naive Bayes
| Supervised Learning
Optimizing machine learning models based on previously observed features and labels. Typically, the goal is to attach the most likely
label to some provided features. §

| Model
An approximation of a relationship between an input and an output. 1
I Heuristic
An approach to finding a solution which Is typically faster but less accurate than some optimal solution. §
| Bernoulli Distribution @
Adistribution which evaluates a particular outcome as binary. In the Bernoulli Nalve Bayes classifier case, a word was either ina |
message or not in a message, which is binary representation. §
| prior @
Indicates the probability of a particular class regardless of the features of some example.
| Likelihood @
The probability of some features given a particular class. §
| Evidence @ |
‘The denominator of the Naive Bayes classifier. §

| Posterior @ |
The probability of a class given some features. §
| Vocabulary 3
The list of words that the Naive Bayes classifier recognizes.
| Laplace smoothing @
Atype of additive smoothing which mitigates the chance of encountering zero probabilities within the Naive Bayes classifier. 3
| Tokenization
The splitting of some raw textual input into individual words or elements.
| Featurization 1
The process of transforming raw inputs into something a model can perform training and predictions on. 1
| Vectorizer 1
Used in a step of featurizing. It transforms some input into something else. One example is a binary vectorizer which transforms
tokenized messages Into a binary vector Indicating which items In the vocabulary appear In the message. §
| Stop Word §
Aword, typically discarded, which doesn't add much predictive value.

| Stemming
Removing the ending modifiers of words, leaving the stem of the word.
| Lemmatization
Amore calculated form of stemming which ensures the proper lemma resuits from removing the word modifiers.

# Evaluation
| Decision Point
Also decision rule or threshold, s a cut-off point in which anything below the cutoff is determined to be a certan class and anything
above the cut-off is the other class. §

I Accuracy 1
The number of true positives plus the number of true negatives divided by the total number of examples. 1

| Unbalanced Classes 1
When one class Is far more frequently observed than another class. §

| Model Training 1
Determining the model parameter values. §

| Confusion Matrix §
In the binary case, a 2x2 matrix indicating the number of true positives, true negatives, false positives and false negatives. 1

| Sensitivity
Also recall, is the proportion of true positives which are correctly classified.

| Specificity
The proportion of true negatives which are correctly classified.

| Precision
~ The number of true positives divided by the true positives plus false positives. ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,j
| F1Score §
The harmonic mean of the precision and recall,
| Validation §
The technique of holding out some portion of examples to be tested separately from the set of examples used to train the model.
| Generalize
The ability of a model to perform well on the test set as well as examples beyond the test set.
| Receiver Operator Characteristic Curve
Also ROC curve, is a plot of how the specificity and sensitivity change as the decision threshold changes. The area under the ROC curve,
or AUC, Is the probability that a randomly chosen positive example will have a higher prediction probability of being positive thana
randomly chosen negative example. §
| Hyperparameter i
Any parameter associated with a model which Is not learned. §

# Naive Bayes Optimization
| Muitinomial Distribution @
Adistribution which models the probability of counts of particular outcomes.
| TF-IDF
Short for Term Frequency-Inverse Document Frequency, TF-IDF is a method of transforming features, usually representing counts ofi
words, into values representing the overall importance of different words across some documents. 1
| Online Learning, 1
Incremental learning within a model to represent an incrementally changing population. 1
I N-gram i
Aserles of adjacent words of length n.
| Feature Hashing i
Representing feature Inputs, such as articles or messages, as the result of hashes modded by predetermined value.
| scikit-learn i
Amachine learning python library which provides implementations of regression, classification, and clustering.
| Kernel Density Estimation (@) |
Also KDE, a way to estimate the probability distribution of some data. 1

# KNearest Neighbors
I TF-IDF i
Short for Term Frequency-Inverse Document Frequency, TF-IDF Is a method of transforming features, usually representing counts of
words, into values representing the overall importance of different words across some documents. §
I Cluster §
A consolidated group of points. §
| Euciidean Distance @|
The length of the line between two points
| Feature Normalization
Typically referring to feature scaling that places the values of a feature between 0 and 1.
| Feature Standardization
Typically referring to feature scaling that centers the values of a feature around the mean of the feature and scales by the standard |
deviation of the feature. i
| Jaccard Distance @ |
One minus the ratio of the number of like binary feature values and the number of like and unlike binary feature values, excluding
Instances of matching zeros. §
| simple Matching Distance @ |
One minus the ratio of the number of like binary feature values and the number of ike and uniike binary feature values. |
| Manhattan Distance @
Sum of the absolute differences of two input features. §
| Hamming Distance @ |
The sum of the non-matching categorical feature values. |

# Decision Tree
| Decision Tree
Atree-based model which traverses examples down to leaf nodes by the properties of the examples features.
| sample Size
The number of observations taken from a complete population.
| Classification and Regression Tree
Also CART, Is an algorithm for constructing an approximate optimal decision tree for given examples. 1
| Missing Data
When some features within an example are missing. i
I Split Point
Apalr of feature and feature value which is assigned to a node In a decision tree. This split point will determine which examples will go
left and which examples go right based on the feature and feature value 1
| Gini Impurity 1
Used as a way to determine the best split point for a given node In a classification tree. It's based on the probability of incorrectly §
classifying an item based on all of the items In the node. §
| Surrogate Split
Asuboptimal split point reserved for examples which are missing the optimal split point feature.
| Mean Squared Error
The average squared difference between the prediction and true label across all examples.
| Boosting
An ensemble technique which trains many weak learners sequentially, with each subsequent weak learner being trained on the
previous weak learner's error. This generally reduces the bias error.
| Bagging
Also bootstrap aggregation, a sampling technique which selects subsets of examples and/or features to train an ensemble of models.
This generally reduces the variance error.
| Weak Learner
Shallow decision trees in our case. However, it generally can be any underfitting model
| Ensemble
Using more than one model to produce a single prediction.
I Random Forest
An ensemble technique which trains many independent weak learners. This generally reduces the variance error.
| xGBoost %
An open-source library which provides a gradient boosted framework. Short for Extreme Gradient Boosting.

| Lightem &
An open-source library created by Microsoft which provides a distributed gradient boosted framework. Short for Light Gradient
Boosted Models.

# Linear Regression
I Line of Best Fit
The line through data points which best describe the relationship of a dependent variable with one or more independent variables,
Ordinary least squares can be used to find the line of best fit
| P-vaiue @
The probability of finding a particular result, or a greater result, given a null hypothesis being true.
| confidence interval @)
The possible range of an unknown value. Often comes with some degree of probability e.g. 95% confidence interval.
| corretation @
The relationship between a dependent and independent variable.
| R-squared @
Also the coefficient of determination, the percent of variance in the dependent variable explained by the independent variable(s).
I Residuals
The distance between points and a particular line.
| Independent Variable
Avariable whose variation is independent from other variables.
| One-hot Encoding
An encoding for categorical variables where every value that a variable can take on is represented as a binary vector
| Dependent Variable
Avarlable whose variation depends on other variables.
| Variance Infiation Factor @)
Ameasure of multicollinearity in a regression model.
| collinearity @)
When one or more (multicollinearity) independent variables are not actually independent.
| Feature Interaction
Features that are multiplied by one another in order to express relationships that can't be represented by adding the independent
variable terms together.
| Nonlinear Regression
Atype of regression which models nonlinear relationships in the independent variables.
| simpson's Paradox
When a pattern emerges In segments of examples but is no longer present when the segments are grouped together.
| statsmodels %

Python module which provides various statistical tools.
| coefficient @

Another name for a parameter in the regression model.

# Logistic Regression
| sigmoid Function
Also the logistic function, a function which outputs a range from 0 to 1
| closed-Form solution @)
For our case, this is what ordinary least squares provides for linear regression. It's a formula which solves an equation.
| Cross-Entropy Loss
Aloss function which is used In dlassification tasks. It's technically the entropy of the true labels plus the KL-divergence of the predicted
and true labels. Minimizing the cross-entropy minimizes the difference between the true and predicted label distributions.
| Parameters
Also, weights, or coefficients. Values to be learned during the model training.
| Learning Rate
Amultiple, typically less than 1, used during the parameter update step during model training to smooth the learning process.
| odds Ratio @)
The degree of associate between two events. If the odds ratio is 1, then the two events are independent. If the odds ratio is greater
than 1, the events are positively correlated, otherwise the events are negatively correlated.
| Multinomial Logistic Regression
Logistic Regression in which there are more than two classes to be predicted across.
I Softmax
Asigmoid which is generalized to more than two classes to be predicted against.
| Gradient Descent
An iterative algorithm with the goal of optimizing some parameters of a given function with respect to some loss function. If done in
batches, all of the examples are considered for an iteration of gradient descent. In mini-batch gradient descent, a subset of examples
are considered for a single Iteration. Stochastic gradient descent considers a single example per gradient descent iteration.
| Downsampling
Removing a number of majority class examples. Typically done in addition to upweighting,
| Upweighting
Increasing the impact a minority class example has on the loss function. Typically done in addition to downsampling.
| Epoch
One complete cycle of training on all of the examples.
| Regularization
Atechnique of limiting the ability for a model to overfit by encouraging the values parameters to take on smaller values.
| Early Stopping
Halting the gradient descent process prior to approaching a minima or maxima.
| Mcfadden's Pseudo R-squared @ |
An analog to linear regression's R-squared which typically takes on smaller values than the traditional R-squared.
| Generative Model
A model which aims to approximate the joint probability of the features and labels.
| Discriminative Model
A model which aims to approximate the conditional probability of the features and labels.

# Support Vector Machine
| Support Vectors
The most difficult to separate points in regard to a decision boundary. They Influence the location and orlentation of the hyperplane.
| Margin
The space between the hyperplane and the support vectors. In the case of soft margin Support Vector Machines, this margin includes.
slack.
| Hyperplane @
A decision boundary in any dimension.
| Norm @
Here, the L2 Norm, is the square root of the sum of squares of each element in a vector.
| Outlier
Afeature or group of features which vary significantly from the other features.
I Slack
The relaxing of the constraint that all examples must lie outside of the margin. This creates a soft-margin SVM.
| Hinge Loss.
Aloss function which Is used by a soft-margin SVM.
| Sub-gradient 1
The gradient of a non-differentiable function. §
| Non.differentiable @ |
Afunction which has kinks in which a derivative is not defined.
| convex Function @
Function with one optima. |
| Kernel Trick
The process of finding the dot product of a high dimensional representation of feature without computing the high dimensional |
representation Itself. A common kernel is the Radial Basis Function kernel. §

# Unsupervised Learning

## KMeans
| Unsupervised Learning 1
An approach within machine learning that takes in unlabeled examples and produces patterns from the provided data. Typically, the
goal s to discover something previously unknown about the unlabeled examples. §
I Centroid
The location of the center of a cluster in n-dimensions.
I Inertia
The sum of distances between each point and the centroid.
| Local optima @
AAmaxima or minima which is not the global optima
| Non-convex Function @
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
| Eigendecomposition @ |
Applicable only to square matrices, the method of factoring a matrix into its eigenvalues and eigenvectors. An eigenvector is a vector |
which applies a linear transformation to some matrix being factored. The eigenvalues scale the eigenvector values. 3
| Principal Component Analysis 1
Also PCA, is eigendecomposition performed on the covariance matrix of some particular data. The eigenvectors then describe the §
principle components and the elgenvalues indicate the variance described by each principal component. Typically, VD Is usedto |
perform PCA. PCA assumes linear correlations in the data. If that assumption Is not true, then you can use kernel PCA. §
| orthogonal @
Perpendicular s n-dimensions.
Note: PCA assumes linear correlations. For non-linear use kernel PCA.

# Deep Learning

# Recommender Systems
| Content Filtering
A recommendation technique which takes into account a single user's features and many items' features.
| Collaborative Filtering
A recommendation technique which uses many user's and item's data typically in the form of a user-item matrix.
| User-item Matrix
/A matrix which contains the Interactions of users and items. Items can take the form of products, music, and videos.
| Pearson Correlation @
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
Pearson distance helps us deal with different kind of users (say optimist and pessimist)

# Ranking
| Candidate Generator 1
A system which outputs the candidates to be ranked. This is sometimes referred to as retrieving the ‘top-k' §
| Embedding Space 1
The n-dimensional space where embeddings exist. Typically, the embedding space can be used to generate the top-k candidates by
using the k-nearest neighbors algorithm. §
| cosine similarity @ |
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



**This package is inspired by this awesoome [repo](https://github.com/eriklindernoren/ML-From-Scratch#implementations)**
