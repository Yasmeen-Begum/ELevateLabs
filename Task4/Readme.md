Breast Cancer Classification Using Logistic Regression
Project Overview

This project builds a binary classification model using logistic regression to distinguish between benign and malignant breast cancer tumors. The dataset used is the Breast Cancer Wisconsin (Diagnostic) Dataset from Scikit-learn. The model is evaluated using common metrics such as confusion matrix, precision, recall, and ROC-AUC.
Technologies Used

    -Python 3

    -NumPy

    -Pandas

    -Matplotlib

    -Scikit-learn

Setup and Installation

    Clone this repository (if applicable):

```
git clone <repository-url>
cd <repository-directory>
```

Install required libraries (if not already installed):

```
pip install numpy pandas matplotlib scikit-learn
```
Step-by-Step Instructions
1. Import Libraries

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
```
2. Load Dataset

Load the Breast Cancer Wisconsin Dataset.

```
data = load_breast_cancer()
X = data.data
y = data.target
```
3. Split Data into Train and Test Sets

Split data into training and testing sets in an 80:20 ratio.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
4. Standardize Features

Standardize features to have mean 0 and variance 1.

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
5. Fit Logistic Regression Model

Train logistic regression model on the training data.

```
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train, y_train)
```
6. Predict and Evaluate

Predict probabilities on test set for threshold tuning and evaluation.

```
y_prob = model.predict_proba(X_test)[:, 1]
y_pred_05 = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred_05)
precision = precision_score(y_test, y_pred_05)
recall = recall_score(y_test, y_pred_05)
roc_auc = roc_auc_score(y_test, y_prob)
```
7. Plot ROC Curve

Visualize the ROC curve for classifier at different thresholds.

```
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()
```
8. Threshold Tuning Example

Evaluate model performance using a different classification threshold.

```
threshold = 0.3
y_pred_tuned = (y_prob >= threshold).astype(int)
cm_tuned = confusion_matrix(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)

print("Confusion Matrix (threshold=0.5):\n", cm)
print(f"Precision (threshold=0.5): {precision:.2f}")
print(f"Recall (threshold=0.5): {recall:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

print("\nConfusion Matrix (threshold=0.3):\n", cm_tuned)
print(f"Precision (threshold=0.3): {precision_tuned:.2f}")
print(f"Recall (threshold=0.3): {recall_tuned:.2f}")
```
Here are detailed answers to the interview questions on logistic regression:

1. **How does logistic regression differ from linear regression?**  
   Logistic regression is used for classification problems where the output is categorical (often binary), predicting the probability that an input belongs to a category. Linear regression predicts continuous numeric outcomes. Logistic regression models the log-odds of the probability using a sigmoid function which constrains output between 0 and 1, whereas linear regression predicts values on an unrestricted continuous scale. Logistic regression uses maximum likelihood estimation while linear regression uses least squares estimation. The underlying assumptions and objectives also differ significantly.

2. **What is the sigmoid function?**  
   The sigmoid function is an S-shaped curve that maps any real-valued number into the range (0, 1), interpreting the output of logistic regression as a probability. It is defined as  
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$  
   This function squashes input values so that they can be interpreted as probabilities.

3. **What is precision vs recall?**  
   - Precision is the ratio of true positive predictions to all positive predictions (true positives + false positives). It measures the accuracy of positive predictions.  
   - Recall (or sensitivity) is the ratio of true positives to all actual positives (true positives + false negatives). It measures how well the model captures all positive instances.

4. **What is the ROC-AUC curve?**  
   The ROC (Receiver Operating Characteristic) curve plots the true positive rate (Recall) against the false positive rate for different classification thresholds. AUC (Area Under the Curve) quantifies the overall ability of the model to discriminate between classes; an AUC of 1 means perfect classification, 0.5 means random guessing.

5. **What is the confusion matrix?**  
   It is a table summarizing the performance of a classification model by showing counts of true positives, true negatives, false positives, and false negatives. It helps evaluate how well the model is classifying each category

6. **What happens if classes are imbalanced?**  
   With class imbalance (one class much larger than the other), accuracy can be misleading because the model might predict the majority class always. Precision, recall, and other metrics like F1-score become more important. Techniques such as resampling, synthetic data generation, or using algorithms robust to imbalance are recommended.

7. **How do you choose the threshold?**  
   The classification threshold (default 0.5) can be tuned based on the trade-off between precision and recall required in the specific application. ROC curves, precision-recall curves, or domain-specific costs of false positives/negatives help decide the appropriate threshold

8. **Can logistic regression be used for multi-class problems?**  
   Yes, logistic regression can be extended to multi-class classification by using strategies such as one-vs-rest (OvR) or multinomial logistic regression, allowing it to predict probabilities for multiple classes rather than just binary outcomes

