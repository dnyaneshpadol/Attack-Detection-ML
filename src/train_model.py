import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def train_evaluate_save(X, y, model_output_path, outputs_dir, test_size=0.2, random_state=42):

    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")

    # 🔥 Improved Model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=random_state,
        n_jobs=-1
    )

    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    # F1 Score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y.unique())

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=labels,
                yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs(outputs_dir, exist_ok=True)
    plt.savefig(f"{outputs_dir}/confusion_matrix.png")
    plt.show()

    # Cross Validation
    print("\nCross Validation (F1 Weighted)...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')

    print(f"CV Scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f}")

    # Feature Importance
    feature_importance = model.feature_importances_
    feature_names = np.array(X.columns)

    top_idx = np.argsort(feature_importance)[-20:][::-1]

    plt.figure(figsize=(10, 8))
    sns.barplot(x=feature_importance[top_idx], y=feature_names[top_idx])
    plt.title("Top 20 Features")

    plt.savefig(f"{outputs_dir}/feature_importance.png")
    plt.show()

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)

    print(f"\nModel saved at: {model_output_path}")

    return model, X_test, y_test