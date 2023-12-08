# Task2.a
import pandas as pd
from IPython.display import HTML, display
import tabulate

# Example: Assuming df is your DataFrame with the dataset
# Replace this with your actual DataFrame
df = pd.DataFrame([
    ["Female", "University", 103000, "Yes"],
    ["Female", "HighSchool", 90500, "No"],
    # ... add all rows here ...
], columns=["Gender", "Education", "Income", "Has_pet"])

# Calculate the likelihood for the 'Gender' feature
likelihood_gender = pd.crosstab(df['Gender'], df['Has_pet'], normalize='index')

# Convert the DataFrame to a format suitable for tabulate
table_data = [(index, *row) for index, row in likelihood_gender.iterrows()]
headers = ["Gender", "Has_pet (No)", "Has_pet (Yes)"]

# Display the table
display(HTML(tabulate.tabulate(table_data, headers=headers, tablefmt='html')))

## Task2.b
# To compute these probabilities, we need to use Bayes' Theorem. The general form of Bayes' Theorem for our case is:

# P(\text{Has_pet} | \text{Feature}) = \frac{P(\text{Feature} | \text{Has_pet}) \times P(\text{Has_pet})}{P(\text{Feature})}

# Where:

# P(\text{Has_pet} | \text{Feature}) is the posterior probability (what we want to calculate).
# P(\text{Feature} | \text{Has_pet}) is the likelihood, which you should have computed in Task 2a.
# P(\text{Has_pet}) is the prior probability of having a pet.
# �
# (
# Feature
# )
# P(Feature) is the probability of the feature.
# Here's how you can implement this in Python, assuming you have a DataFrame df containing your data:

# Calculate Prior Probabilities:
# These are simply the probabilities of the classes ('Has_pet' being 'Yes' or 'No').

# Calculate Marginal Probabilities of the Features:
# These are probabilities of observing each feature value regardless of the class.

# Calculate Posterior Probabilities:
# Apply Bayes' Theorem using the likelihoods from Task 2a, and the priors and marginals calculated in steps 1 and 2.
import pandas as pd

# Example DataFrame
# Replace this with your actual DataFrame
df = pd.DataFrame([
    ["Female", "University", 103000, "Yes"],
    ["Female", "HighSchool", 90500, "No"],
    # ... add all rows here ...
], columns=["Gender", "Education", "Income", "Has_pet"])

# Calculate the prior probabilities
prior_yes = df['Has_pet'].value_counts(normalize=True)['Yes']
prior_no = df['Has_pet'].value_counts(normalize=True)['No']

# Calculate the marginal probabilities for Gender
marginal_male = df['Gender'].value_counts(normalize=True)['Male']
marginal_female = df['Gender'].value_counts(normalize=True)['Female']

# Calculate the likelihoods (from Task 2a)
likelihood_male_no = pd.crosstab(df['Gender'], df['Has_pet'], normalize='index').at['Male', 'No']
likelihood_female_yes = pd.crosstab(df['Gender'], df['Has_pet'], normalize='index').at['Female', 'Yes']

# Calculate the posterior probabilities
posterior_no_given_male = (likelihood_male_no * prior_no) / marginal_male
posterior_yes_given_female = (likelihood_female_yes * prior_yes) / marginal_female

print("P(No | Male):", posterior_no_given_male)
print("P(Yes | Female):", posterior_yes_given_female)


## Task2.c
#> Steps for Implementation:
# Calculate Mean and Standard Deviation:
# For each class ('Has_pet' being Yes or No), calculate the mean (μ) and standard deviation (σ) of the 'Income' feature.

#> Use the Normal Distribution Formula:
# Plug these values into the normal distribution formula to find the probability density of a specific income value for each class.

#> Compute Likelihoods for a Specific Income Value:
# Calculate the likelihood of having a pet given a specific income (e.g., 90000).

import pandas as pd
import numpy as np
from scipy.stats import norm

# Example DataFrame
# Replace this with your actual DataFrame
df = pd.DataFrame([
    ["Female", "University", 103000, "Yes"],
    ["Female", "HighSchool", 90500, "No"],
    # ... add all rows here ...
], columns=["Gender", "Education", "Income", "Has_pet"])

# Function to calculate likelihood using Normal Distribution
def calculate_likelihood(df, feature, target_value, class_label):
    # Filter the DataFrame for the given class
    class_df = df[df['Has_pet'] == class_label]
    
    # Calculate mean and standard deviation for the feature
    mean = class_df[feature].mean()
    std = class_df[feature].std()

    # Calculate the probability density for the target value
    probability_density = norm.pdf(target_value, mean, std)
    return probability_density

# Compute likelihood for Income = 90000 given 'Yes' and 'No'
likelihood_income_yes = calculate_likelihood(df, 'Income', 90000, 'Yes')
likelihood_income_no = calculate_likelihood(df, 'Income', 90000, 'No')

print("P(Income = 90000 | Yes):", likelihood_income_yes)
print("P(Income = 90000 | No):", likelihood_income_no)


## Task2.d

# Calculate Posterior Probability for Each Class:
# For each class ('Yes' and 'No'), calculate the posterior probability using the Naïve Bayes formula. This involves multiplying the likelihoods of each feature given the class, and the prior probability of the class, then normalizing these probabilities.

# Combine Likelihoods for Each Feature:
# Since Naïve Bayes assumes feature independence, you can simply multiply the likelihoods of individual features.

# Normalize the Probabilities:
# The probabilities should sum up to 1 across all classes.

# Function to calculate posterior probability
def calculate_posterior(df, feature_values, class_label):
    # Initialize posterior probability with the prior probability
    posterior = df['Has_pet'].value_counts(normalize=True)[class_label]
    
    # Multiply with the likelihoods of each feature
    for feature, value in feature_values.items():
        likelihood = calculate_likelihood(df, feature, value, class_label)
        posterior *= likelihood

    return posterior

# Feature values for the predictions
features_1 = {'Education': 'University', 'Gender': 'Female', 'Income': 100000}
features_2 = {'Education': 'HighSchool', 'Gender': 'Male', 'Income': 92000}

# Calculate posterior probabilities for 'Yes' and 'No' for the first scenario
posterior_yes_1 = calculate_posterior(df, features_1, 'Yes')
posterior_no_1 = calculate_posterior(df, features_1, 'No')

# Normalize probabilities
total_1 = posterior_yes_1 + posterior_no_1
posterior_yes_1 /= total_1
posterior_no_1 /= total_1

# Calculate posterior probabilities for 'Yes' and 'No' for the second scenario
posterior_yes_2 = calculate_posterior(df, features_2, 'Yes')
posterior_no_2 = calculate_posterior(df, features_2, 'No')

# Normalize probabilities
total_2 = posterior_yes_2 + posterior_no_2
posterior_yes_2 /= total_2
posterior_no_2 /= total_2

print("Prediction for (Education=University, Gender=Female, Income=100000):")
print("P(Yes):", posterior_yes_1, "P(No):", posterior_no_1)

print("\nPrediction for (Education=HighSchool, Gender=Male, Income=92000):")
print("P(Yes):", posterior_yes_2, "P(No):", posterior_no_2)


## Task2.e
# For Task 2e, you'll implement a Naïve Bayes Classifier to perform classification on the Iris dataset. The Iris dataset is a famous dataset that contains measurements of iris flowers and is often used for testing machine learning algorithms. The dataset has four numerical features (sepal length, sepal width, petal length, and petal width) and a target variable that classifies the flowers into one of three species.

# Since the Iris dataset only contains numerical features, you can use the GaussianNB classifier from scikit-learn, which assumes that the likelihood of the features is Gaussian.

# Steps to Implement Naïve Bayes on the Iris Dataset:
# Load the Iris Dataset: Import the dataset from scikit-learn.

# Split the Dataset: Divide the dataset into training and testing sets.

# Create and Train the Naïve Bayes Model: Use the GaussianNB model.

# Make Predictions and Evaluate the Model: Use the testing set to evaluate the model's performance.
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the model
gnb.fit(X_train, y_train)

# Make predictions
y_pred = gnb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Model Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", conf_matrix)
