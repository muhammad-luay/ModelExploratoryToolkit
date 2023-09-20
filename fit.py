import warnings
import time
from sklearn.utils import all_estimators
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Create a synthetic dataset
X, y = make_classification(n_samples=10000, n_features=200, n_informative=20, n_redundant=20, 
                           n_repeated=0, n_classes=2, n_clusters_per_class=4, 
                           flip_y=0.01, class_sep=0.7)

percentages = [99.99, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]
table = []

# Get all classifier estimators from sklearn
estimators = all_estimators(type_filter='classifier')

for percentage in percentages:
    _, X_sample, _, y_sample = train_test_split(X, y, test_size=percentage/100, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.7, random_state=42)

    results = []
    start_time = time.time()  # Start timing the script

    for name, ClassifierClass in estimators:
        try:
            classifier = ClassifierClass()          # instantiate the classifier
            classifier.fit(X_train, y_train)        # train the classifier
            y_pred = classifier.predict(X_test)     # make predictions on the test set
            accuracy = accuracy_score(y_test, y_pred) # compute accuracy
            results.append((name, accuracy))          # append results to the list
        except Exception as e:
            pass

    end_time = time.time()  # End timing the script
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    entry = {
        'Percentage': percentage,
        'Results': sorted_results,
        'Runtime': end_time - start_time
    }

    table.append(entry)



# Create an empty DataFrame to store the results
columns = ['Percentage of data used', 'Runtime in Seconds']
for i in range(1, len(estimators) + 1):
    columns.extend([f'Classifier {i}', f'Accuracy {i}'])

df_results = pd.DataFrame(columns=columns)

# Fill the DataFrame with results
for entry in table:
    percentage = entry['Percentage']
    runtime = entry['Runtime']
    
    data = {'Percentage of data used': percentage, 'Runtime in Seconds': runtime}
    for idx, (classifier, accuracy) in enumerate(entry['Results'], 1):
        data[f'Classifier {idx}'] = classifier
        data[f'Accuracy {idx}'] = accuracy

    df_results = df_results.append(data, ignore_index=True)

# Display the DataFrame
df_results.iloc[:, :10]
