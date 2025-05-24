# ------------------------
# General information
# ------------------------

print("\nNumber of samples:", len(dataset_dict["X"]))
print("\nInfo variabiles:")
print(dataset_dict["variables"])

# ------------------------
# Analyzing missing value 
# ------------------------

import pandas as pd
import matplotlib.pyplot as plt

total_missing = dataset_dict["X"].isnull().sum().sum()
missing_per_column = dataset_dict["X"].isnull().sum()
missing_percentage = dataset_dict["X"].isnull().mean() * 100

missing_data = pd.DataFrame({
    'Missing Values': missing_per_column,
    'Missing Percentage': missing_percentage
})

missing_data = missing_data[missing_data['Missing Values'] > 0]

print(missing_data)

plt.figure(figsize=(10, 6))
missing_data['Missing Percentage'].sort_values().plot(kind='barh', color='skyblue')
plt.title('Percentage of Missing Values per Column')
plt.xlabel('Missing Percentage (%)')
plt.ylabel('Columns')
plt.show()

# ------------------------------
# Handle missing values
# ------------------------------

# 1. Remove columns with more than 50% missing values
threshold = 50
columns_to_drop = missing_data[missing_data['Missing Percentage'] > threshold].index
dataset_dict_cleaned = dataset_dict["X"].drop(columns=columns_to_drop)

print(f"\nColumns removed: {columns_to_drop.tolist()}")

# 2. Impute remaining missing values with mode
for column in dataset_dict_cleaned.columns:
    if dataset_dict_cleaned[column].isnull().sum() > 0:
        mode_value = dataset_dict_cleaned[column].mode()[0] 
        dataset_dict_cleaned[column].fillna(mode_value, inplace=True)
        print(f"Imputed missing values in column: {column} with mode value: {mode_value}")

total_missing_after = dataset_dict_cleaned.isnull().sum().sum()
print(f"\nNumber of total missing values: {total_missing_after}")
dataset_dict["X"] = dataset_dict_cleaned

# ------------------------
# Statistical Analysis - numeric columns
# ------------------------


print("Statistical Analysis for numeric columns:\n") 
dataset_dict["X"] = dataset_dict["X"].drop(columns="Unnamed: 0", errors="ignore")
print(dataset_dict["X"].describe().round(1))

	
# ------------------------
# Distribution of the categorical variables
# ------------------------


import matplotlib.pyplot as plt
import seaborn as sns
import math

categorical_cols = dataset_dict["X"].select_dtypes(include='object').columns

n_cols = 3
n_rows = math.ceil(len(categorical_cols) / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten()

for i, col in enumerate(categorical_cols):
    sns.countplot(data=dataset_dict["X"], x=col, order=dataset_dict["X"][col].value_counts().index, ax=axes[i])
    axes[i].set_title(f'Distribution: {col}')
    axes[i].tick_params(axis='x')

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# ------------------------
# Distribution of the numeric variables
# ------------------------

import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = dataset_dict["X"].select_dtypes(include=['int64', 'float64']).columns

fig, axes = plt.subplots(1, len(numeric_cols), figsize=(5 * len(numeric_cols), 4))

for ax, col in zip(axes, numeric_cols):
    sns.histplot(dataset_dict["X"][col], kde=True, ax=ax)
    ax.set_title(f'{col}')
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.suptitle('Distribution of numeric variables', fontsize=16)
plt.tight_layout()
plt.show()

# ------------------------
# Correlation Matrix - numeric columns
# ------------------------

correlation_matrix = dataset_dict["X"].corr()

print("Correlation Matrix:\n")
print(correlation_matrix)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# ------------------------
# Correlation Matrix - converting categorical into numeric value
# ------------------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_encoded = dataset_dict["X"].apply(pd.Categorical).apply(lambda x: x.cat.codes)
df_encoded['target'] = dataset_dict["y"]['class'].astype('category').cat.codes

corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'correlation'})
plt.title('Correlation Matrix')
plt.show()
