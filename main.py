# ------------------------
# Main Funciton
# ------------------------
from sklearn.preprocessing import LabelEncoder

def main():
    dataset = fetch_dataset()

    X = pd.get_dummies(dataset["X"])

    y = LabelEncoder().fit_transform(dataset["y"].values.ravel())

    return {
        "X": X,
        "y": y,
        "variables": dataset["variables"],
        "feature_names": X.columns.tolist()
    }


# ------------------------
# Main Funciton
# ------------------------

# Load Dataset
dataset_dict = fetch_dataset()
dataset_dict["X"].head()
