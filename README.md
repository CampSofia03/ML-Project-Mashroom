
# 🧠🤖 ML-Project: Tree predictors for binary classification 🍄🍄

Predicting the Edibility of Mushrooms: A Classification Task to Distinguish Between Edible and Poisonous Varieties Using a Classification Rule.

## 📦 Installing Requirements

Before you start experimenting with the code, you'll need to install the required dependencies. To do so, simply run:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

All data-related things are in `data.py`. To view data stats, use

```[bash]
python data.py
```

The dataset paper is at https://shorturl.at/3SLVm


## ℹ️ General Info

The project performs data inspection, handles missing values, and provides statistical summaries and visualizations of both categorical and numerical features to understand the dataset before modeling.

## 🌲 Tree

A custom decision tree classifier was implemented from scratch, supporting Gini, Entropy, and Scaled Entropy as split criteria.
The model includes training, evaluation (accuracy, confusion matrix), and visualization. Hyperparameter tuning explores depth and splitting criteria to optimize performance and control overfitting.
