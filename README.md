# imp-emeritus-pcmlai-portfolio-project: Credit Card Fraud Detection
Credit Card Fraud Detection

## Overview

This project focuses on credit card fraud detection using a dataset obtained from Kaggle. The dataset contains credit card transactions made by European cardholders in September 2013, presenting a binary classification challenge for fraud detection.

## Dataset Information

- **Dataset Name:** Credit Card Fraud Detection
- **Source:** Kaggle ([Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud))
- **Context:** The dataset consists of credit card transactions made by European cardholders in September 2013, posing a binary classification challenge for fraud detection.

The dataset contains 284,807 transactions with 31 features, including anonymized features resulting from PCA transformation for sensitive information protection. The features include:

- Time elapsed between the transaction and the first one in the dataset (in seconds)
- Anonymized PCA-transformed features (V1-V28)
- Transaction amount
- Binary target variable indicating fraud (1) or non-fraud (0)

The dataset is highly imbalanced, with a small percentage of fraud transactions.

## Data Preprocessing

The data preprocessing steps applied to the dataset are as follows:

- **Handling Missing Values:** No missing values were observed in the dataset.
- **Feature Scaling:** The 'Amount' and 'Time' features were scaled for uniformity.
- **Feature Engineering:** PCA was applied to the anonymized features (V1-V28) to reduce dimensionality.
- **Imbalanced Data Handling:** Strategies such as oversampling, undersampling, or synthetic data generation can be used to address the class imbalance.

## Exploration and Analysis

During the exploration and analysis phase, the following tasks were performed:

- **Distribution of Classes:** The analysis revealed that the majority of transactions are non-fraudulent, resulting in a class imbalance.
- **Correlation Analysis:** Correlations between features were evaluated to understand their impact on the target variable.

## Model Evaluation

The performance of various machine learning models was evaluated using the F1 score metric. The F1 scores for each model are as follows:


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

**Model Comparison:**

1. **Random Forest:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.93
   - **Recall:** 0.80
   - **F1-Score:** 0.86
   - **Comments:** The Random Forest model demonstrates outstanding performance across key metrics, making it a strong candidate for promotion.

2. **Gradient Boosting:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.64
   - **Recall:** 0.16
   - **F1-Score:** 0.26
   - **Comments:** While achieving high accuracy, the model's lower recall indicates room for improvement, especially in identifying positive cases.

3. **XGBoost:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.92
   - **Recall:** 0.71
   - **F1-Score:** 0.80
   - **Comments:** XGBoost performs well with a balanced precision and recall, making it a strong contender for promotion.

4. **Ensemble:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.10
   - **Recall:** 0.01
   - **F1-Score:** 0.02
   - **Comments:** The ensemble model shows challenges in correctly identifying positive cases, requiring further investigation and improvement.

5. **LightGBM:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.15
   - **Recall:** 0.35
   - **F1-Score:** 0.21
   - **Comments:** LightGBM exhibits lower precision and recall, suggesting the need for enhancement, particularly in identifying positive cases.

6. **CatBoost:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.93
   - **Recall:** 0.78
   - **F1-Score:** 0.84
   - **Comments:** CatBoost demonstrates a good performance with high precision and recall, making it a strong candidate for promotion.

7. **Neural Network:**
   - **Accuracy Score:** 1.0
   - **Precision:** 0.96
   - **Recall:** 0.69
   - **F1-Score:** 0.80
   - **Comments:** The neural network model shows high accuracy but with room for improvement in recall, especially for positive cases.

8. **Autoencoder:**
   - **Accuracy Score:** 0.0
   - **Precision:** 0.00
   - **Recall:** 1.00
   - **F1-Score:** 0.00
   - **Comments:** The autoencoder model presents challenges, reflected in a low accuracy score and precision, indicating the need for reconsideration or improvement.

9. **Clustering:**
   - **Accuracy Score:** 0.45
   - **Precision:** 0.00
   - **Recall:** 0.61
   - **F1-Score:** 0.00
   - **Comments:** Clustering demonstrates limitations in precision and F1-Score, suggesting a need for further refinement or an alternative approach.

10. **Naive Bayes:**
    - **Accuracy Score:** 0.98
    - **Precision:** 0.06
    - **Recall:** 0.85
    - **F1-Score:** 0.11
    - **Comments:** While achieving high accuracy, Naive Bayes shows challenges in precision and F1-Score, indicating potential areas for improvement.

11. **K-Nearest Neighbors:**
    - **Accuracy Score:** 1.0
    - **Precision:** 0.94
    - **Recall:** 0.76
    - **F1-Score:** 0.84
    - **Comments:** K-Nearest Neighbors performs well with balanced precision and recall, making it a strong candidate for promotion.

12. **SVM:**
    - **Accuracy Score:** 1.0
    - **Precision:** 0.92
    - **Recall:** 0.70
    - **F1-Score:** 0.80
    - **Comments:** SVM demonstrates good performance, particularly in precision and accuracy, making it a potential candidate for promotion.

13. **Decision Trees:**
    - **Accuracy Score:** 1.0
    - **Precision:** 0.74
    - **Recall:** 0.71
    - **F1-Score:** 0.73
    - **Comments:** Decision Trees show balanced performance, suggesting suitability for the task, but with room for improvement.

14. **Logistic Regression:**
    - **Accuracy Score:** 1.0
    - **Precision:** 0.85
    - **Recall:** 0.56
    - **F1-Score:** 0.67
    - **Comments:** Logistic Regression exhibits good precision but may benefit from improvement in recall.

15. **Isolation Forest:**
    - **Accuracy Score:** 0.0
    - **Precision:** 0.00
    - **Recall:** 0.66
    - **F1-Score:** 0.00
    - **Comments:** Isolation Forest presents challenges, reflected in a low accuracy score and precision, indicating the need for reconsideration or improvement.
	
## Hyperparameter Optimization

To improve the performance of models with lower F1 scores, hyperparameter tuning was performed using Bayesian Optimization. This technique helps to find the optimal set of hyperparameters for a given model.

## Recommendations

Based on the analysis and evaluation, the following recommendations are provided:

1. **Model Performance:**
   - Evaluate the models based on key metrics such as accuracy, precision, recall, and F1 Score.
   - Identify models with high performance and those requiring improvement.

2. **Hyperparameter Tuning:**
   - Continue hyperparameter tuning for models with lower performance, especially for those with poor recall or F1 Score.

3. **Feature Engineering:**
   - Investigate the possibility of additional feature engineering to enhance the models' ability to capture relevant patterns in the data.

4. **Model-Specific Recommendations:**
   - Tailor recommendations based on the characteristics of each model. For instance, focus on improving the autoencoder's architecture or reconsider the choice of clustering for this classification task.

5. **Further Evaluation:**
   - Consider additional evaluation metrics, such as area under the ROC curve (AUC-ROC), to gain a comprehensive understanding of model performance.

Remember to validate these recommendations through rigorous experimentation, and continually iterate on the models and strategies to achieve optimal results.

## Project Contributors

The project was contributed to by the following individuals:

- **Data Collection:** Kaggle user [akingbhenry]
- **Data Cleaning and Analysis:** [Henry Akingbemisilu]
- **Model Development and Optimization:** [Henry Akingbemisilu]
- **Documentation and Presentation:** [Henry Akingbemisilu]

## Version History

- **Version 1.0 (1 December 2023):** Initial dataset exploration and model development.
- **Version 1.1 (5 December 2023):** Incorporation of feedback, model optimization, and final recommendations.

This README.md provides comprehensive information about the project, including dataset details, preprocessing steps, model evaluation, hyperparameter optimization, and recommendations.
