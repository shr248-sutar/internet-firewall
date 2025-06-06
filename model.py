import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    # Load dataset
    df = pd.read_csv("log2_trimmed.csv")

    # Remove rows with invalid Action values
    df = df[df['Action'].isin(['allow', 'deny'])]

    # Convert Action column to binary: allow = 0, deny = 1
    df['Action'] = df['Action'].map({'allow': 0, 'deny': 1})

    # Features and target
    X = df.drop("Action", axis=1)
    y = df["Action"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return clf, accuracy
