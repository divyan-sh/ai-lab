## Task2.a

> Open Jupyter Notebook: Start Jupyter Notebook in your environment. This can typically be done by opening a command line or terminal and typing jupyter notebook. Your default web browser should open with the Jupyter interface.

> Open Your Notebook: Navigate to and open the Lab3_task2.ipynb file within the Jupyter interface.

## Task2.b

> To compute these probabilities, we need to use Bayes' Theorem. The general form of Bayes' Theorem for our case is:
> P(\text{Has_pet} | \text{Feature}) = \frac{P(\text{Feature} | \text{Has_pet}) \times P(\text{Has_pet})}{P(\text{Feature})}

Here's how you can implement this in Python, assuming you have a DataFrame df containing your data:

Calculate Prior Probabilities:
These are simply the probabilities of the classes ('Has_pet' being 'Yes' or 'No').

Calculate Marginal Probabilities of the Features:
These are probabilities of observing each feature value regardless of the class.

Calculate Posterior Probabilities:
Apply Bayes' Theorem using the likelihoods from Task 2a, and the priors and marginals calculated in steps 1 and 2.
