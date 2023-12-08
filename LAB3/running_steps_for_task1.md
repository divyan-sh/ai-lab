# Steps to Proceed with Task 1 Implementation and Environment Setup

1. Environment Setup:

   > Conda Environment setup: https://medium.com/@nrk25693/how-to-add-your-conda-environment-to-your-jupyter-notebook-in-just-4-steps-abeab8b8d084
   > bnlearn instalation: https://erdogant.github.io/bnlearn/pages/html/Quickstart.html

2. ipynb file: https://saturncloud.io/blog/what-is-the-ipynb-jupyter-notebook-file-extension-and-how-to-open-it/#:~:text=The%20ipynb%20file%20extension%20stands,%2C%20visualizations%2C%20and%20narrative%20text.

3. Task1 code implementation:
   > add pandas library to your conda environment. pip install pandas.
   > follow the code mentioned in lab3_task1.py

To proceed with Task 1 in your local environment, please ensure that bnlearn is installed. Follow these steps:

Install Required Libraries:
Open your terminal or command prompt and execute the following commands:

bash
Copy code
pip install xlrd pandas
pip install bnlearn
This will install the necessary libraries, including pandas for data handling and bnlearn for Bayesian Network modeling.

Load and Preprocess the Dataset:
Load the smart grid dataset using pandas and preprocess it by converting categorical columns to type 'category'. This is crucial for the bnlearn library to correctly handle the data.

Structure Learning:
Use bnlearn to learn the structure of the Bayesian Network from the dataset. This involves creating a Directed Acyclic Graph (DAG) that represents the dependencies between variables.

Parameter Learning:
After obtaining the structure, estimate the (conditional) probability distributions of the individual variables. This step involves fitting the data to the learned structure.

Performing Inference:
Once the network is constructed, you can perform inference to find the probabilities for specific conditions as outlined in Task 1.

Run the Notebook:
Execute each cell in the Lab3_task1.ipynb notebook in your local Jupyter environment. Make sure to modify the inference steps to align with the specific conditions of the smart grid dataset.

Remember to ensure that your Jupyter environment is properly set up to use the bnlearn library. If you're using a Conda environment, follow the guide I mentioned earlier to integrate it with Jupyter Notebook.

> https://saturncloud.io/blog/what-is-the-ipynb-jupyter-notebook-file-extension-and-how-to-open-it/#:~:text=The%20ipynb%20file%20extension%20stands,%2C%20visualizations%2C%20and%20narrative%20text.
