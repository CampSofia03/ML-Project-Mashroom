import os
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
import time


# --------------------
# Decision Tree Classes
# --------------------

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    
    def __init__(self, max_depth=None, min_samples_split=2, entropy_threshold=None,
                 max_leaf_nodes=None, split_function='gini', feature_names=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.entropy_threshold = entropy_threshold
        self.max_leaf_nodes = max_leaf_nodes
        self.feature_names = feature_names
        self.root = None
        self.leaf_count = 0

        if split_function == 'gini':
            self.criterion_func = self.gini
        elif split_function == 'entropy':
            self.criterion_func = self.entropy
        elif split_function == 'scaled_entropy':
            self.criterion_func = self.scaled_entropy
        else:
            raise ValueError("Unsupported criterion")


    def fit(self, X, y):
        self.root = self.grow_tree(X, y)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])

    def predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(x, node.left)
        return self.predict_one(x, node.right)

    def grow_tree(self, X, y, depth=0):
        if (len(set(y)) == 1 or
            len(y) < self.min_samples_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            (self.entropy_threshold is not None and self.criterion_func(y) < self.entropy_threshold) or
            (self.max_leaf_nodes is not None and self.leaf_count >= self.max_leaf_nodes)):
            return TreeNode(value=self.most_common(y))

        best_feat, best_thresh = self.best_split(X, y)
        if best_feat is None:
            return TreeNode(value=self.most_common(y))

        self.leaf_count += 1
        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx
        left = self.grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.grow_tree(X[right_idx], y[right_idx], depth + 1)

        return TreeNode(feature=best_feat, threshold=best_thresh, left=left, right=right)

    def best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_idx = X[:, feature] <= thresh
                right_idx = ~left_idx
                if len(y[left_idx]) == 0 or len(y[right_idx]) == 0:
                    continue
                gain = self.information_gain(y, y[left_idx], y[right_idx])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feature, thresh
        return best_feat, best_thresh

    def information_gain(self, parent, left, right):
        weight_l = len(left) / len(parent)
        weight_r = len(right) / len(parent)
        return self.criterion_func(parent) - (weight_l * self.criterion_func(left) + weight_r * self.criterion_func(right))

    def most_common(self, y):
        return np.bincount(y).argmax()

    def gini(self, y):
        probs = np.bincount(y) / len(y)
        return 1 - np.sum(probs ** 2)

    def entropy(self, y):
        probs = np.bincount(y) / len(y)
        return -sum(p * np.log2(p + 1e-9) for p in probs if p > 0)

    def scaled_entropy(self, y):
        probs = np.bincount(y) / len(y)
        return -sum((p / 2) * np.log2(p + 1e-9) for p in probs if p > 0)

    def visualize(self):
        dot = Digraph()
        self.visualize_tree(self.root, dot)
        return dot

    def visualize_tree(self, node, dot, parent_id=None, edge_label=""):
        current_id = str(id(node))
    
        if node.is_leaf():
            label = f"Predict: {node.value}"
            dot.node(current_id, label, shape="ellipse", style="filled", fillcolor="lightgreen")
        else:
            name = self.feature_names[node.feature] if self.feature_names else f"X[{node.feature}]"
            label = f"{name} <= {node.threshold}"
            dot.node(current_id, label, shape="box", style="filled", fillcolor="lightblue")
        if parent_id is not None:
            dot.edge(parent_id, current_id, label=edge_label)
        if node.left:
            self.visualize_tree(node.left, dot, current_id, "True")
        if node.right:
            self.visualize_tree(node.right, dot, current_id, "False")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():

    dataset = fetch_dataset()

    if "Unnamed: 0" in dataset["X"].columns:
        dataset["X"] = dataset["X"].drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in dataset["y"].columns:
        dataset["y"] = dataset["y"].drop(columns=["Unnamed: 0"])

    # Feature and Target
    X_raw = dataset["X"]
    y_raw = dataset["y"]

    # One-hot encoding - feature
    X = pd.get_dummies(X_raw)

    # Label encoding
    y = LabelEncoder().fit_transform(y_raw.values.ravel().astype(str))

    # Training and test set
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train = X_train_raw.copy()
    X_test = X_test_raw.reindex(columns=X_train.columns, fill_value=0)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X_train.columns.tolist()
    }
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

if __name__ == "__main__":
    data = main()
    X_train = data["X_train"].values
    X_test = data["X_test"].values
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    for criterion in ['gini', 'entropy', 'scaled_entropy']:
        print(f"\nUsing split function: {criterion}")
        tree_model = DecisionTree(
            max_depth=5,
            min_samples_split=5,
            entropy_threshold=0.01,
            split_function=criterion,
            feature_names=feature_names
        )
        
        # Training
        start_time = time.time()
        tree_model.fit(X_train, y_train)
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} sec")

        # Assessment
        y_pred = tree_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Zero-One Loss: {np.mean(y_pred != y_test):.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {criterion}")
        plt.tight_layout()
        plt.show()

        # Tree visualization
        tree_graph = tree_model.visualize()
        tree_graph.render(f"tree_visual_{criterion}", format="png", view=True)

# ------------------------
# Hyperparameter Tuning
# ------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

split_criteria = ['gini', 'entropy', 'scaled_entropy']
depth_range = [2, 3, 4, 5, 6, 7, 8, 9]

results = []

for criterion in split_criteria:
    for depth in depth_range:
        #Model Preparation
        tree = DecisionTree(
            max_depth=depth,
            min_samples_split=5,
            entropy_threshold=0.01,
            split_function=criterion,
            feature_names=feature_names
        )

        #Training
        tree.fit(X_train, y_train)

        #Assessment
        y_train_pred = tree.predict(X_train)
        y_test_pred = tree.predict(X_test)
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        results.append({
            "Criterion": criterion,
            "Max Depth": depth,
            "Train Accuracy": train_acc,
            "Test Accuracy": test_acc,
            "Overfitting Gap": train_acc - test_acc
        })


results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by=["Criterion", "Max Depth"])
print(results_df.to_string(index=False))

#Visualization
pivot = results_df.pivot(index="Criterion", columns="Max Depth", values="Test Accuracy")

plt.figure(figsize=(8, 5))
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Test Accuracy - Depth vs Criterion")
plt.xlabel("Max Depth")
plt.ylabel("Split Criterion")
plt.tight_layout()
plt.show()

import pandas as pd

results_df["Overfitting Gap"] = results_df["Train Accuracy"] - results_df["Test Accuracy"]

#Best - test accuracy
best_test_model = results_df.loc[results_df["Test Accuracy"].idxmax()]

#Best - overfitting gap
best_balanced_model = results_df.loc[results_df["Overfitting Gap"].abs().idxmin()]

print("Best model - Test Accuracy max:")
print(best_test_model)
print("\nBalanced model - min overfitting gap:")
print(best_balanced_model)
