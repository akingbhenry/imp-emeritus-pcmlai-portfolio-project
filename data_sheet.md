# Data Sheet: Credit Card Fraud Detection

## Overview

- **Dataset Name:** Credit Card Fraud Detection
- **Source:** Kaggle (Credit Card Fraud Detection - Kaggle)
- **Context:** This dataset comprises credit card transactions made by European cardholders in September 2013, posing a binary classification challenge for fraud detection.

## Dataset Information

- **Features:**
  - **Time:** Time elapsed between the transaction and the first one in the dataset (in seconds).
  - **V1, V2, ..., V28:** Anonymized features resulting from PCA transformation for sensitive information protection.
  - **Amount:** Transaction amount.
  - **Class:** Binary target variable indicating fraud (1) or non-fraud (0).

- **Instances:** 284,807 transactions
- **Features:** 31 features (Time, V1-V28, Amount, Class)
- **Target Variable:** Class (Binary: 0 for non-fraud, 1 for fraud)
- **Imbalance:** Highly imbalanced dataset with a small percentage of fraud transactions.

## Data Preprocessing

- **Handling Missing Values:** No missing values observed.
- **Duplicate Check**: Dataset does not have any duplicates
- **Feature Scaling:** 'Amount' and 'Time' features scaled for uniformity.
- **Feature Engineering:** PCA applied to anonymized features (V1-V28) for dimensionality reduction.
- **Imbalanced Data Handling:** Strategies like oversampling, undersampling, or synthetic data use may address class imbalance.

## Exploration and Analysis

- **Distribution of Classes:** Majority of transactions are non-fraudulent, leading to class imbalance.
- **Correlation Analysis:** Evaluate correlations between features and explore impact on the target variable.

## Model Evaluation

- **F1 Scores:**
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

## Recommendations

As things stand, the Random Forest model appears to be the most promising choice for promotion. Here are the reasons:

1. **High Accuracy:** The Random Forest model achieved an accuracy score of 1.0, indicating perfect predictions on the given dataset.

2. **Balanced Metrics:** The model exhibits a good balance between precision, recall, and F1 Score for both classes (0 and 1). This indicates that the model is effective in correctly identifying both normal and fraudulent transactions.

3. **High Precision and Recall for Class 1:** Specifically, for the fraud class (class 1), Random Forest achieved a precision of 0.93 and a recall of 0.80. This implies that the model is effective at both identifying fraud cases and minimizing false positives.

4. **Robustness:** Random Forest models are known for their robustness and resistance to overfitting. They perform well on a variety of datasets and are less prone to being influenced by noise.

5. **Interpretability:** Random Forest models provide feature importances, making it easier to interpret and understand the factors contributing to predictions. This interpretability is crucial for gaining insights into the underlying patterns in the data.

6. **Ensemble Nature:** Random Forest is an ensemble of decision trees, which generally leads to improved generalization and performance over individual decision trees.

7. **Suitability for Classification Tasks:** Random Forest is well-suited for binary classification tasks, making it a strong candidate for fraud detection.

However, it's important to note that the choice of the best model depends on various factors, including the specific requirements of the application, computational resources, and the cost associated with false positives and false negatives. It is advisable to thoroughly validate the model's performance on unseen data and consider potential deployment constraints before finalizing the promotion decision. Additionally, monitoring the model's performance in a production environment is crucial to ensure continued effectiveness over time.

## Overall Evaluation:
1. **Model Performance:**
   - Evaluate the models based on key metrics such as accuracy, precision, recall, and F1 Score.
   - Identify models with high performance and those requiring improvement.

2. **Hyperparameter Tuning:**
   - Continue hyperparameter tuning for models with lower performance, especially for those with poor recall or F1 Score.

3. **Feature Engineering:**
   - Investigate the possibility of additional feature engineering to enhance the models' ability to capture relevant patterns in the data.

4. **Model-Specific Recommendations:**
   - Tailor recommendations based on the characteristics of each model. For instance, focus on improving the autoencoder's architecture or reconsider the choice of clustering for this classification task.


Remember to validate these recommendations through rigorous experimentation, and continually iterate on the models and strategies to achieve optimal results.

## Project Contributors

- **Data Collection:** Kaggle user [akingbhenry]
- **Data Cleaning and Analysis:** [Henry Akingbemisilu]
- **Model Development and Optimization:** [Henry Akingbemisilu]
- **Documentation and Presentation:** [Henry Akingbemisilu]

## Version History

- **Version 1.0 (1 December 2023):** Initial dataset exploration and model development.
- **Version 1.1 (5 December 2023):** Incorporation of feedback, model optimization, and final recommendations.

This data sheet comprehensively outlines the project, providing a valuable resource for understanding the dataset, preprocessing steps, model evaluation, and recommendations.
