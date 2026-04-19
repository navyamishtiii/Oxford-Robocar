from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

def run_ml_models(df):
    X = df[[
    "mean_intensity", 
    "std_intensity", 
    "skewness",
    "kurtosis", 
    "clutter_index",
    "temporal_variance"
]]
    y = df["label"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n🔹 {name}")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        results[name] = model

        # Feature importance (only RF)
        if name == "Random Forest":
            plt.figure()
            importances = model.feature_importances_
            plt.barh(X.columns, importances)
            plt.title("Feature Importance (Random Forest)")
            plt.show()

    return results, scaler