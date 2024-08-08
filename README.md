## Binary Prediction of Smoker Status using Bio-Signals

The repository  holds attempt to apply machine algorithms to Binary Prediction of Smoker Status using Bio-Signals using data from kaggle. https://www.kaggle.com/competitions/playground-series-s3e24/data

## This section could contain a short paragraph which include the following:

Definition of the tasks / challenge: The challenge is defined by kaggle and its objective is to apply machine learning to predict the smoking status of an individual based on bio-signals. Machine learning will enable us to see different factors associated with an smoker. The data is tabular(a comma separated file) with 29.2 MB. The dataset provides Information about different individuals. Information like age, height in cm weight in kg.. etc The machine task is supervised specifically binary classification.

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

<img width="865" alt="Screen Shot 2024-08-05 at 4 51 55 PM" src="https://github.com/user-attachments/assets/d12eaeca-3e15-46fe-b8fe-b678fdd532fa">

<img width="918" alt="Screen Shot 2024-08-05 at 4 52 39 PM" src="https://github.com/user-attachments/assets/d4506392-38d0-4f7e-a64c-3a212e17dc23">

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
# Performance Comparison
The metrics used to evaluate the model performance  are F-1 score, precision, recall and accuracy. 
Show/compare results in one table.
<img width="407" alt="Screen Shot 2024-08-05 at 5 22 38 PM" src="https://github.com/user-attachments/assets/7d2267e2-ab0f-4392-841e-beea4b2c3cd6">

ROC curves.
<img width="696" alt="Screen Shot 2024-08-05 at 5 25 51 PM" src="https://github.com/user-attachments/assets/220b020d-167d-41ad-af69-9b97cb731cf3">

<img width="601" alt="Screen Shot 2024-08-05 at 5 26 32 PM" src="https://github.com/user-attachments/assets/fc84b657-6752-4f61-8adc-f5cb163a40aa">


# Conclusions

The exploratory data analysis (EDA) revealed valuable insights into the dataset, such as the distribution of age, height, and weight, 
and the most promising features for predicting smoking status. Data preprocessing steps like handling outliers and normalizing numerical 
features were performed to improve model performance.

 Both the Decision Tree and XGBoost models showed good performance on the test set, achieving high accuracy, precision, recall, and F1-score. 
 The XGBoost model slightly outperformed the Decision Tree, indicating its potential for more accurate predictions.

The confusion matrices and ROC curves provided further insights into the models' performance. The XGBoost model demonstrated a higher 
area under the ROC curve (AUC), suggesting its superior ability to discriminate between smokers and non-smokers.

Overall, the machine learning models developed in this project can be valuable tools for predicting smoking status based on individual 
characteristics. Further improvements can be explored by fine-tuning model parameters, incorporating additional features, or experimenting 
with other advanced machine learning algorithms.
# Future Work
- Hyperparameter tuning for both models.
- Feature engineering to create more informative features.
- Trying other algorithms like logistic regression or support vector machines.

Studies that can be done from here are:
Ensemble Methods:Combine predictions from multiple models (e.g., Decision Tree, XGBoost, Logistic Regression) using techniques like bagging or boosting to create a more robust and accurate model.
External Validation: If possible, validate the model on an independent dataset to assess its generalizability and performance on unseen data.

# How to reproduce results
 - Set up an  environment
 - Prepare your data
 - Train the model
 - Evaluate the model

# List all relavent files and describe their role in the package.
Data_exploration_chall.ipyng-Loads and explores the dataset getting information about the dataset.
Data_visualization_chall-Creates visualizations in order to better understand  the data.
Data_Preprocessing_chall-Performs cleaning in the dataset, prepares the data for machine learning.
Machine_learning_chall-Creates the models and evaluate  its perfomance. 

# Software Setup
Library: numpy, scikit learn, pandas, matplotlib,spicy stats.
# Data
https://www.kaggle.com/competitions/playground-series-s3e24/data

# Citations
https://scikit-learn.org/stable/modules/tree.html#decision-trees
https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
https://www.kaggle.com/code/dansbecker/xgboost
https://www.w3schools.com/python/python_ml_auc_roc.asp
