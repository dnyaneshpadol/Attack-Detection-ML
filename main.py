import argparse
from pathlib import Path
import pandas as pd

def main():
    """Main pipeline: load → EDA → preprocess → train → evaluate → save → predict."""
    
    # Project paths
    project_root = Path(__file__).parent
    dataset_dir = project_root / "dataset"
    model_dir = project_root / "model"
    outputs_dir = project_root / "outputs"
    
    model_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    train_file = dataset_dir / "KDDTrain+.txt"
    test_file = dataset_dir / "KDDTest+.txt"
    model_file = model_dir / "rf_model.joblib"

    # Argument parser
    parser = argparse.ArgumentParser(description="Attack Detection ML Pipeline")
    parser.add_argument("--train", action="store_true", default=True, help="Train model")
    parser.add_argument("--predict-only", action="store_true", help="Load model and predict on test sample")
    args = parser.parse_args()

    # =========================
    # PREDICT ONLY MODE
    # =========================
    if args.predict_only:
        from src.preprocessing import load_nsl_kdd, prepare_data
        from src.predict import load_model, predict_single

        print("Loading model...")
        model = load_model(str(model_file))

        print("Loading test data...")
        test_df = load_nsl_kdd(str(test_file))
        X_test, y_test = prepare_data(test_df)

        print("\n" + "="*60)
        print("SAMPLE PREDICTION")
        print("="*60)

        for i in range(min(5, len(X_test))):
            sample = X_test.iloc[i:i+1]
            true_label = y_test.iloc[i]
            prediction = predict_single(model, sample)

            print(f"\nSample {i+1}")
            print(f"Actual Label   : {true_label}")
            print(f"Predicted Label: {prediction}")
            print("-"*40)

    # =========================
    # FULL PIPELINE
    # =========================
    else:
        from src.preprocessing import load_nsl_kdd, exploratory_data_analysis, plot_class_distribution, prepare_data
        from src.train_model import train_evaluate_save
        from src.predict import load_model, predict_single
        from sklearn.metrics import accuracy_score, classification_report

        # 1. Load data
        print("Loading training dataset...")
        if not train_file.exists():
            print("Training file not found!")
            return

        train_df = load_nsl_kdd(str(train_file))

        # 2. EDA
        exploratory_data_analysis(train_df)
        plot_class_distribution(train_df, str(outputs_dir))

        # 3. Preprocessing
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)

        X, y = prepare_data(train_df)
        print(f"Preprocessed feature shape: {X.shape}")
        print(f"Label distribution:\n{pd.Series(y).value_counts()}")

        # 4. Train, evaluate, save
        model, X_test, y_test = train_evaluate_save(
            X, y, str(model_file), str(outputs_dir),
            test_size=0.2, random_state=42
        )

        # 5. Accuracy & Report
        y_pred = model.predict(X_test)

        print("\n" + "="*60)
        print("MODEL PERFORMANCE")
        print("="*60)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # 6. Sample Predictions
        print("\n" + "="*60)
        print("SAMPLE PREDICTIONS")
        print("="*60)

        for i in range(min(5, len(X_test))):
            sample = X_test.iloc[i:i+1]
            true_label = y_test.iloc[i]
            prediction = predict_single(model, sample)

            print(f"\nSample {i+1}")
            print(f"Actual Label   : {true_label}")
            print(f"Predicted Label: {prediction}")
            print("-"*40)

        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    main()