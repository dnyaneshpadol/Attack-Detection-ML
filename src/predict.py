import joblib
import pandas as pd


def load_model(path):
    """Load trained model."""
    return joblib.load(path)


def format_label(label):
    """Convert model output into readable format."""
    if label == "normal":
        return "Normal Traffic"
    else:
        return f"Attack Detected ({label})"


def predict_traffic(model, X_sample):
    """
    Predict traffic type and return readable output.
    """
    predictions = model.predict(X_sample)
    output = [format_label(pred) for pred in predictions]
    return output


def predict_single(model, X_sample):
    """Predict single sample."""
    result = predict_traffic(model, X_sample)
    return result[0] if result else "Unknown"