# Titanic Dataset Preprocessing and Analysis

## Overview

This project demonstrates essential data cleaning, preprocessing, and exploratory data analysis steps on the Titanic dataset using Python. The workflow walks through handling missing values, encoding categorical variables, feature scaling, outlier detection/removal, and basic visualization, aiming to prepare the data for predictive modeling.

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Steps

### 1. Importing Libraries

All necessary libraries for data manipulation, visualization, and preprocessing are imported, including scikit-learn utilities for imputation, scaling, and encoding tasks.

### 2. Loading the Dataset

The Titanic data is loaded from a CSV file:
```python
df = pd.read_csv('/content/Titanic-Dataset.csv')
```

### 3. Initial Data Exploration

View the first entries and summarize data types and missing values:
```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```
Helps identify columns requiring cleaning and understand dataset shape.[5][1]

### 4. Handling Missing Values

- **Numerical Columns (e.g., Age):** Missing values filled with median:
  ```python
  num_imputer = SimpleImputer(strategy='median')
  df['Age'] = num_imputer.fit_transform(df[['Age']])
  ```
- **Categorical Columns (e.g., Embarked):** Mode used to fill missing values:
  ```python
  cat_imputer=SimpleImputer(strategy='most_frequent')
  df['Embarked']=cat_imputer.fit_transform(df[['Embarked']])[:,0]
  ```

### 5. Encoding Categorical Variables

- **Label Encoding:** For binary columns like 'Sex':
  ```python
  l = LabelEncoder()
  df['Sex'] = l.fit_transform(df['Sex'])
  ```
- **One-Hot Encoding:** For multicategory columns like 'Embarked':
  ```python
  df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
  ```

### 6. Feature Scaling

'Age' and 'Fare' columns are scaled using `MinMaxScaler` to normalize them between 0 and 1:
```python
scaler = MinMaxScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
```

### 7. Outlier Detection and Removal

- Visualization using a boxplot;
- Removal with Interquartile Range (IQR) filtering:
  ```python
  Q1 = df['Fare'].quantile(0.25)
  Q2 = df['Fare'].quantile(0.75)
  IQR = Q2 - Q1
  df = df[~((df['Fare'] < (Q1 - 1.5*IQR)) | (df['Fare'] > (Q2 + 1.5*IQR)))]
  ```

### 8. Saving Cleaned Data

The cleaned dataset is saved as `titanic_cleaned.csv`:
```python
df.to_csv('titanic_cleaned.csv', index=False)
```

### 9. Visualization

Pairplots and other exploratory plots showcase feature relationships.

## Usage

1. Clone this repository and download the Titanic dataset (e.g., from Kaggle).
2. Run the Jupyter Notebook or corresponding Python script.
3. The final cleaned dataset will be stored as `titanic_cleaned.csv`.

***

[10](https://codesignal.com/learn/courses/data-cleaning-and-preprocessing-in-machine-learning/lessons/data-preprocessing-the-titanic-dataset-exploration)
