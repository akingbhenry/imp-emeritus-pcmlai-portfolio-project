# Model Card: Credit Card Fraud Detection

## Model Information

- **Model Name:** FraudDetectNet
- **Model Type:** Ensemble Model
- **Purpose:** The model aims to detect fraudulent credit card transactions based on features such as transaction amount, time, and anonymized principal components obtained through PCA.

## Training Data

- **Dataset:** Credit Card Fraud Detection - Kaggle
- **Dataset Source:** [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Dataset Description:** The dataset contains transactions made by credit cards in September 2013 by European cardholders. It presents a binary classification challenge for fraud detection.

## Model Training

- **Features:**
  - **Time:** Time elapsed between this transaction and the first transaction in the dataset (in seconds).
  - **V1, V2, ..., V28:** Anonymized features resulting from a PCA transformation.
  - **Amount:** Transaction amount.

- **Target Variable:** Class (Binary: 0 for non-fraud, 1 for fraud)
- **Training Objective:** Minimize misclassifications, with a focus on maximizing the F1 score due to the imbalanced nature of the dataset.

- **Training Techniques:**
  - Hyperparameter tuning using Bayesian optimization.
  - Learning with Autoencoder, Unsupervised Learning (Clustering), Isolation Forest.
  - Principal Component Analysis (PCA) for dimensionality reduction.

## Model Performance

- **Evaluation Metric:** F1 Score

	| Model                    | F1 Score |
	|--------------------------|----------|
	| Logistic Regression      | 0.67     |
	| Decision Trees           | 0.73     |
	| Random Forests           | 0.86     |
	| Support Vector Machines  | 0.80     |
	| Gradient Boosting        | 0.26     |
	| Naive Bayes              | 0.11     |
	| K-Nearest Neighbors      | 0.84     |
	| XGBoost                  | 0.80     |
	| Isolation Forest         | 0.00     |
	| Ensemble Models          | 0.02     |
	| LightGBM and CatBoost    | 0.84     |
	| Unsupervised Learning    | 0.00     |
	| Autoencoder              | 0.00     |
	| Neural Networks          | 0.80     |


## Handling Imbalanced Data

  - Employed oversampling and undersampling techniques to address the class imbalance.

## Usage Guidelines

- **Intended Use:**
  - The model is intended for use in real-time credit card transaction processing systems.
  - It is designed to identify potentially fraudulent transactions.

- **Potential Users:**
  - Fraud detection teams within financial institutions.
  - Credit card payment processors.

- **Potential Biases:**
  - The model's performance may be influenced by changes in transaction patterns or evolving fraud tactics.

## Limitations

- The model assumes stationarity in transaction patterns and may require periodic retraining to adapt to changing conditions.

## Model Outputs

- **Output Type:** Binary Classification (0 for non-fraud, 1 for fraud)

- **Model Uncertainty:**
  - The model does not provide uncertainty estimates for its predictions.

## Ethical Considerations

- The model is designed to minimize false positives to prevent blocking legitimate transactions.
- Regular monitoring and evaluation are necessary to identify and address potential biases and inaccuracies.

## Model Version History

- Version 1.0: Initial model release (1 December 2023).
- Version 1.1: Incorporation of additional training data and model optimization (5 December 2023).

This model card provides a comprehensive overview of the credit card fraud detection model, covering its purpose, training data, performance metrics, usage guidelines, potential biases, and ethical considerations.
