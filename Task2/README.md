Titanic Dataset Exploratory Data Analysis (EDA)
Project Overview

This project performs exploratory data analysis on the Titanic passenger dataset to uncover patterns, trends, and relationships between features relevant to passenger survival. The analysis uses descriptive statistics and visualizations to gain insights into the data.

## Step 1: Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

**Explanation:**  
These libraries are essential for data manipulation (pandas, numpy) and visualization (seaborn, matplotlib). Seaborn is built on matplotlib but provides simpler syntax and nicer visuals.

***

## Step 2: Load Dataset

```python
df = pd.read_csv('/content/Titanic-Dataset.csv')
```

**Explanation:**  
Loads the Titanic dataset from the specified CSV path into a pandas DataFrame named `df`. This DataFrame allows you to easily inspect, clean, and analyze the data.

***

## Step 3: Summary Statistics for Numeric Features

```python
print("Summary statistics for numeric features")
print(df.describe())
```

**Explanation:**  
`df.describe()` outputs statistical summaries including count, mean, std deviation, min, max, and quartile values for numeric columns like Age and Fare. This gives insight into distribution and variability.

***

## Step 4: Histograms for Numeric Features

```python
df.hist(column=['Age','Fare'], bins=20, figsize=(12,5), edgecolor='black')
plt.suptitle('Histograms of Age and Fare')
plt.show()
```

**Explanation:**  
Histograms visually display the frequency distribution of Age and Fare. Bins divide value ranges into intervals to show how data is distributed (e.g., is Age skewed older or younger?).

***

## Step 5: Boxplots Grouped by Survival

```python
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.boxplot(x='Survived', y='Age', data=df)
plt.title('Age distribution by Survival')

plt.subplot(1,2,2)
sns.boxplot(x='Survived', y='Fare', data=df)
plt.title('Fare distribution by Survival')
plt.tight_layout()
plt.show()
```

**Explanation:**  
Boxplots reveal the spread and quartiles of Age and Fare grouped by Survived (0 = did not survive, 1 = survived). Helps identify differences in distribution and potential outliers between survivor groups.

***

## Step 6: Pairplot for Numerical Features by Survival

```python
sns.pairplot(df, vars=['Age','Fare'], hue='Survived', palette='Set1')
plt.suptitle('Pairplot: Age and Fare by Survival', y=1.02)
plt.show()
```

**Explanation:**  
Pairplot shows scatterplots and distributions of Age and Fare with Survival as color distinction. It explores potential patterns or clusters differentiated by survival status.

***

## Step 7: Correlation Matrix Heatmap

```python
corr = df[['Survived','Pclass','Age','SibSp','Parch','Fare']].corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
```

**Explanation:**  
Heatmap displays correlation coefficients between numeric features and survival. Positive correlation means variables tend to increase together; negative means inverse relation. Useful for identifying predictive features.

***

## Step 8: Countplots for Categorical Features by Survival

```python
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival count by passenger class')

plt.subplot(1,3,2)
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survival count by sex')

plt.subplot(1,3,3)
sns.countplot(x='Embarked', hue='Survived', data=df)
plt.title('Survival count by embarkation point')

plt.tight_layout()
plt.show()
```

**Explanation:**  
Countplots show how many passengers in each category survived or not, helping visualize survival disparities across class, gender, and boarding location.

***

## Step 9: Feature-Level Inferences

```python
print("Average Age of Passengers:", df['Age'].mean())
print("Survival Rate:", df['Survived'].mean())

print("Survival rate by Gender:")
print(df.groupby('Sex')['Survived'].mean())

print("Survival rate by Passenger Class:")
print(df.groupby('Pclass')['Survived'].mean())
```

**Explanation:**  
Basic aggregation to calculate mean age and survival rates overall, and broken down by gender and passenger class to interpret which groups had better chances of survival.

***

[7](https://www.youtube.com/watch?v=Ea_KAcdv1vs)
