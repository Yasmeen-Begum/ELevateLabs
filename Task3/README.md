


# Housing Price Prediction using Linear Regression

This project demonstrates building a multiple linear regression model to predict housing prices based on various features like area, bedrooms, bathrooms, and amenities.

## Dataset

The dataset used contains the following columns:

`price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus`

You can download the dataset (Housing.csv) from this link:  
[Housing Price Dataset]([https://example.com/path-to-your-dataset](https://github.com/Yasmeen-Begum/ELevateLabs/blob/main/Task3/Housing.csv))  
(Replace with actual download URL or your own dataset location)

## Project Setup

1. Clone this repository or download source code files.

2. Make sure Python 3.x is installed along with these libraries:  
   - pandas  
   - numpy  
   - scikit-learn  
   - matplotlib  

You can install the libraries using pip:

```
pip install pandas numpy scikit-learn matplotlib
```

3. Place the dataset file `Housing.csv` in your working directory or update the file path in the code.

## How to Run

1. Import all necessary libraries:

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
```

2. Load the dataset and check for missing values:

```
df = pd.read_csv('/content/Housing.csv')
print(df.isnull().sum())
```

3. Convert categorical columns to dummy variables:

```
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Retain only columns that exist in the dataset 
categorical_cols = [col for col in categorical_cols if col in df.columns]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

4. Define features and target and split data:

```
x = df.drop('price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

5. Train the Linear Regression model and make predictions:

```
model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
```

6. Evaluate the model:

```
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
```

7. Visualize feature coefficients and actual vs predicted prices:

```
features = x.columns
coefficients = model.coef_

plt.figure(figsize=(10,6))
plt.barh(features, coefficients)
plt.xlabel('Coefficient Value')
plt.title('Feature Coefficients')
plt.show()

plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.show()
```






-Interview Question

1. What assumptions does linear regression make?  
- Linearity: The relationship between predictors and the target is linear.  
- Normality: The residuals (errors) are normally distributed.  
- Homoscedasticity: Constant variance of residuals across all levels of predictors.  
- Independence: Observations and residuals are independent of each other.  
- No multicollinearity: Predictors should not be highly correlated with each other.  


2. How do you interpret the coefficients?  
Each coefficient measures the expected change in the target variable for a one-unit increase in that predictor, holding other variables constant. Positive coefficients indicate a direct relationship; negative coefficients indicate an inverse relationship.

3. What is R² score and its significance?  
R² (coefficient of determination) represents the proportion of variance in the dependent variable explained by the independent variables. It ranges from 0 to 1; higher values indicate better model fit and explanatory power.

4. When would you prefer MSE over MAE?  
MSE penalizes larger errors more heavily due to squaring, which is useful when large deviations are particularly undesirable. MAE treats all errors linearly and is more robust to outliers. Prefer MSE when emphasizing larger errors is important.

5. How do you detect multicollinearity?  
Using correlation matrices, Variance Inflation Factor (VIF) scores, or condition indices helps detect multicollinearity. High correlation or high VIF values (>5 or >10) indicate problematic multicollinearity.

6. What is the difference between simple and multiple regression?  
Simple regression uses one predictor to model the target. Multiple regression uses two or more predictors, allowing better explanation of variation in the target by accounting for multiple factors.

7. Can linear regression be used for classification?  
No. Linear regression predicts continuous outcomes, not discrete categories. Classification requires algorithms like logistic regression or decision trees.

8. What happens if you violate regression assumptions?  
Violations can lead to biased or inefficient estimates, invalid hypothesis tests, incorrect confidence intervals, and poor predictions. For example, non-normal errors affect inference validity; heteroscedasticity can bias standard errors; autocorrelation violates independence assumption.


[7](https://en.wikipedia.org/wiki/Linear_regression)
[8](https://analystprep.com/cfa-level-1-exam/quantitative-methods/assumptions-underlying-linear-regression-2/)
[9](https://www.geeksforgeeks.org/machine-learning/python-linear-regression-using-sklearn/)
