# Bernoulli Naive Bayes : It assumes that all our features are binary such that they take only two values. Means 0s can represent “word does not occur in the document” and 1s as "word occurs in the document" .

# Multinomial Naive Bayes : Its is used when we have discrete data (e.g. movie ratings ranging 1 and 5 as each rating will have certain frequency to represent). In text learning we have the count of each word to predict the class or label.

# Gaussian Naive Bayes : Because of the assumption of the normal distribution, Gaussian Naive Bayes is used in cases when all our features are continuous. For example in Iris dataset features are sepal width, petal width, sepal length, petal length. So its features can have different values in data set as width and length can vary. We can’t represent features in terms of their occurrences. This means data is continuous. Hence we use Gaussian Naive Bayes here.

from collections import defaultdict
import math

class MultinomialNB:
    def __init__(self, articles_per_tag):
        self.alpha = 1
        self.priors_per_tag = {}
        self.likelihood_per_word_per_tag = {}
        self.articles_per_tag = articles_per_tag
        self.tags = articles_per_tag.keys()
        self.train()

    def train(self):
        tag_counts_map = {tag: len(self.articles_per_tag[tag]) for tag in self.tags}
        self.priors_per_tag = {tag: tag_counts_map[tag] / sum(tag_counts_map.values()) for tag in self.tags}
        self.likelihood_per_word_per_tag = self.__get_word_likelihoods_per_tag()

    def predict(self, article):
        posteriors_per_tag = {tag: math.log(prior) for tag, prior in self.priors_per_tag.items()}
        for word in article:
            for tag in self.tags:
                posteriors_per_tag[tag] = posteriors_per_tag[tag] + math.log(
                    self.likelihood_per_word_per_tag[word][tag]
                )
        return posteriors_per_tag
                

    def __get_word_likelihoods_per_tag(self):
        word_frequencies_per_tag = defaultdict(lambda: {tag: 0 for tag in self.tags})
        total_word_count_per_tag = defaultdict(int)
        for tag in self.tags:
            for article in self.articles_per_tag[tag]:
                for word in article:
                    word_frequencies_per_tag[word][tag] += 1
                    total_word_count_per_tag[tag] += 1
        word_likelihoods_per_tag = defaultdict(lambda: {tag: 0.5 for tag in self.tags})
        for word, tags_map in word_frequencies_per_tag.items():
            for tag in tags_map.keys():
                word_likelihoods_per_tag[word][tag] = (word_frequencies_per_tag[word][tag] + 1 * self.alpha) / (
                    total_word_count_per_tag[tag] + 2 * self.alpha
                )
        return word_likelihoods_per_tag
        
            
