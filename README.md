## Binary Prediction of Smoker Status using Bio-Signals

The repository  holds attempt to apply machine algorithms to Binary Prediction of Smoker Status using Bio-Signals using data from kaggle. https://www.kaggle.com/competitions/playground-series-s3e24/data

## This section could contain a short paragraph which include the following:

Definition of the tasks / challenge: The challenge is defined by kaggle and its objective is to apply machine learning to predict the smoking status of an individual based on bio-signals. Machine learning will enable to see different factors associated with an smoker. The data is tabular(a comma separated file) with 29.2 MB. The dataset provides Information about different individuals. Information like age, height in cm weight in kg.. etc The machine task is supervised specifically binary classification.
Your approach : The approach in this repository formulates the problem as classification task , using decision tree and XGBoost as the model with the full time series of features as input. We compared the performance of the 2 different  models. 
Summary of the performance achieved: The metrics used to evaluate the model performance  were F-1 score, recall, accuracy and roc curve. The best model was able to predict the smoking status of individuals. The ROC curve for XGBoost was closer to the top-left corner, indicating a stronger ability to discriminate between smokers and non-smokers. 

# Summary of Workdone

# Data
- Data:
 - Type: 
Input: CSV file of features, output: smoker/non-smoker flag in 1st column.
Size: 29.2 MB, 159256 rows and 24 columns
Instances (Train, Test, Validation Split): how many data points? : 82248 individuals  training, 27417 for testing, 27417 for validation
# Preprocessing / Clean up
The data did not require much cleaning, because there no duplicates no missing values. The only cleaning  performed was the removal of outliers. I used the Z-score method to check and handle any outlier. 

# Data Visualization
Show a few visualization of the data and say a few words about what you see.

<img width="638" alt="Screen Shot 2024-08-05 at 4 03 40 PM" src="https://github.com/user-attachments/assets/3ef7bd72-6e29-4e6c-b40c-6fd37cb9315f">

![Screen Shot 2024-07-24 at 11.22.59 PM.png](attachment:fd0f7b55-4a94-47c7-b572-0e9864fa18de.png)

![Screen Shot 2024-07-24 at 11.24.26 PM.png](attachment:f71a2f68-9b95-49ac-ba26-20dd5479af45.png)

<img width="636" alt="Screen Shot 2024-08-05 at 4 05 24 PM" src="https://github.com/user-attachments/assets/3678d221-aa33-4a2d-9564-0f37229809c6">


The  2 and the 3 images is the histogram of some features based on the target. We can see the difference between the smokers and non-smokers.
Theses histograms helps us see the most prosmising features for Machine Learning. 

The first and the forth histograns are plots of categorical values. The urine protein histogram shows the three different  urine categories. 

# Problem Formulation
Define:
Input 
Train_set.csv: A CSV file containing the training data.
Test.csv: A CSV file containing the test data.
Output
submission.csv: A CSV file containing the predicted target values for the test data.

# Models
Decision Tree:
Simple to understand and interpret, can capture non-linear relationships.
Prone to overfitting, can be unstable (small changes in data can lead to different trees).
Achieved a reasonable accuracy on the test set, but might benefit from techniques to reduce overfitting (like pruning).

XGBoost:
High accuracy, handles complex relationships, robust to overfitting due to regularization.
More complex to interpret than decision trees, requires careful tuning of hyperparameters.
Generally outperformed the decision tree in terms of accuracy, precision, recall, and F1-score. The ROC curve also indicates strong discriminatory power.

I tried decision tree model due to its simplicity and interpretability and it served as a starting point to understand the data and the problem before using a complex model. The XGBoost is a gradient boosting algorithm that is known for its high accuracy and efficiency. In this case, we used XGBoost because we wanted to achieve the best possible performance on the smoking prediction task. XGBoost is a good choice for this task because it can handle complex relationships between the features and the target variable. It is also robust to overfitting, which is a common problem with decision trees

# Training
Describe the training:
I used the xgb.train function to train the XGBoost model. I specified the parameters, training data (dtrain), number of boosting rounds, 
Evaluation sets (evallist) for monitoring performance on both training and validation data, and early stopping to prevent overfitting:
Any difficulties? How did you resolve them?
Fortunately, there was no difficulties while training. 
Performance Comparison
Clearly define the key performance metric(s).
Show/compare results in one table.
Show one (or few) visualization(s) of results, for example ROC curves.
Conclusions
State any conclusions you can infer from your work. Example: LSTM work better than GRU.
Future Work
What would be the next thing that you would try.
What are some other studies that can be done starting from here.
How to reproduce results
In this section, provide instructions at least one of the following:
Reproduce your results fully, including training.
Apply this package to other data. For example, how to use the model you trained.
Use this package to perform their own study.
Also describe what resources to use for this package, if appropirate. For example, point them to Collab and TPUs.
Overview of files in repository
Describe the directory structure, if any.

List all relavent files and describe their role in the package.

An example:

utils.py: various functions that are used in cleaning and visualizing data.
preprocess.ipynb: Takes input data in CSV and writes out data frame after cleanup.
visualization.ipynb: Creates various visualizations of the data.
models.py: Contains functions that build the various models.
training-model-1.ipynb: Trains the first model and saves model during training.
training-model-2.ipynb: Trains the second model and saves model during training.
training-model-3.ipynb: Trains the third model and saves model during training.
performance.ipynb: loads multiple trained models and compares results.
inference.ipynb: loads a trained model and applies it to test data to create kaggle submission.
Note that all of these notebooks should contain enough text for someone to understand what is happening.

Software Setup
List all of the required packages.
If not standard, provide or point to instruction for installing the packages.
Describe how to install your package.
Data
Point to where they can download the data.
Lead them through preprocessing steps, if necessary.
Training
Describe how to train the model
Performance Evaluation
Describe how to run the performance evaluation.
Citations
Provide any references.
