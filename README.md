# Overview
This repository contains a script to systematically evaluate performance of all available classifier models provided by Scikit-learn on a synthetic dataset. The idea is to diversify our machine learning model selection process by initially testing all available models, and then picking the best ones for hyperparameter tuning.

The script creates a dataset of 10,000 samples and 200 features. In each iteration, it uses varying percentages of the dataset to train and test the models, ranging from 99.99% to 5%.

## How to Use
1. Ensure you have all the necessary libraries installed:

```pip install scikit-learn pandas```

2. Clone the repository and navigate to its directory.

Run the script:

```python fit.py```

The script will output a DataFrame showcasing the top-performing classifiers for each percentage of data used.

## Key Concepts
Choosing the Right Model: Importance of selecting the right model for the dataset at hand.
Systematic Evaluation: Instead of sticking to popular models, this script evaluates performance across all available models in Scikit-learn.
Data Subset: To save on computational time, the script uses varying percentages of data, showing consistent results even with a fraction of the data.

## Findings
From the experiment conducted using the script, it's evident that even with a fraction of the data, some models consistently outperform others. The resulting DataFrame provides insights into top-performing models across different data subsets, guiding the user to make informed decisions about which models to further explore and fine-tune.

## Conclusion
Diversifying our approach and taking an exploratory stance at the beginning might save us from settling prematurely on a particular model. By exploring and experimenting, we can truly elevate our machine learning model selection process and improve overall performance.


