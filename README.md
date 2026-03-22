# Attack Detection System using Machine Learning

## Project Overview
This project is a Machine Learning-based Intrusion Detection System (IDS) that detects whether network traffic is Normal or an Attack. The system uses the Random Forest algorithm to classify network traffic. The model is trained using the KDD Cup 99 dataset, which is a well-known dataset used for intrusion detection.

## Objective
The objective of this project is to:
- Detect network intrusions using Machine Learning
- Classify network traffic as Normal or Attack
- Evaluate model performance
- Visualize the results using graphs

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## Project Structure
Attack-Detection-ML
│
├── dataset          -> Training and testing dataset
├── src              -> Source code (preprocessing, training, prediction)
├── model            -> Saved trained model
├── outputs          -> Output graphs and results
├── main.py          -> Main program file
├── requirements.txt -> Required Python libraries
└── README.md        -> Project documentation

## Machine Learning Model
This project uses the Random Forest algorithm, which is an ensemble learning method that uses multiple decision trees to improve prediction accuracy.

## Output Results
The project generates the following outputs:
- Confusion Matrix
- Feature Importance Graph
- Class Distribution Graph

## How to Run the Project
1. Install Python
2. Install required libraries:
   pip install -r requirements.txt
3. Train the model:
   python src/train_model.py
4. Run prediction:
   python src/predict.py
5. Run main file:
   python main.py

## Dataset
KDD Cup 99 Intrusion Detection Dataset

## Conclusion
This project demonstrates how Machine Learning can be used in Cyber Security to detect network attacks.

## Author
Dnyaneshwar Padol
