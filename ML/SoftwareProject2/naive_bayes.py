import numpy as np


class CustomNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.likelihoods = {}
        self.classes = None

    def fit(self, X_train, y_train):
        """
        Fit the Naive Bayes classifier according to X (features) and y (target).
        """
        # Ensure that X_train is an integer array
        X_train = X_train.astype(int)

        # Identify the unique classes in the training set
        self.classes = np.unique(y_train)

        # Calculate prior probabilities and likelihoods for each class
        for cls in self.classes:
            # Filter the rows where the target class equals the current class
            X_train_cls = X_train[y_train == cls]

            # Calculate the prior probability for the class
            self.priors[cls] = X_train_cls.shape[0] / X_train.shape[0]

            # Initialize likelihoods for this class
            self.likelihoods[cls] = {}

            # Calculate likelihoods for each feature in this class
            for i in range(X_train.shape[1]):
                feature_values_cls = X_train_cls[:, i]
                probabilities = self.calculate_likelihoods(feature_values_cls)
                self.likelihoods[cls][i] = probabilities

    def predict(self, X_test):
        """
        Perform classification on an array of test vectors X.

        Returns:
        array-like: Predicted classes for each instance in X_test
        """
        predictions = []
        for x in X_test:
            # Calculate posterior probability for each instance in X_test
            class_probabilities = self.calculate_posteriors(x)

            # Choose the class with the highest probability
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            predictions.append(predicted_class)
        return predictions

    def predict_proba(self, X):
        """Calculate the probability of each class for each instance in X.

        Returns:
            np.array: A 2D array of shape (n_samples, n_classes) where each
                      row represents the probability distribution across classes
                      for a given sample.
        """
        probabilities = []
        for x in X:
            # Calculate posterior probabilities for each class
            class_probs = self.calculate_posteriors(x)
            # Ensure that class_probs is a dictionary with a probability for each class
            # If some class probability is missing, it should default to a very small value
            probs = [class_probs.get(cls, 1e-6) for cls in self.classes]

            # Normalize the probabilities so they sum up to 1
            prob_sum = sum(probs)
            normalized_probs = [p / prob_sum for p in probs]

            probabilities.append(normalized_probs)

        return np.array(probabilities)

    @staticmethod
    def calculate_likelihoods(feature_values):
        """
        Calculate the likelihoods of each unique value in a feature.

        Returns:
        dict: Likelihoods for each unique value in the feature
        """
        total_count = len(feature_values)
        value_counts = np.bincount(feature_values)
        likelihoods = {val: (count / total_count) for val, count in enumerate(value_counts)}
        return likelihoods

    def calculate_posteriors(self, x):
        """
        Calculate the posterior probability for each class based on the input features.

        Returns:
        dict: Posterior probabilities for each class
        """
        posteriors = {}
        for cls in self.classes:
            # Start with the prior probability of the class
            prior = self.priors[cls]
            likelihood = 1

            # Multiply with the likelihood of each feature value
            for i, feature_value in enumerate(x):
                likelihood *= self.likelihoods[cls][i].get(feature_value,
                                                           1e-6)  # Using a small value to avoid zero probability

            # Calculate the posterior probability
            posteriors[cls] = prior * likelihood

        return posteriors
