# InsightAI Fraud Detection Model

This project presents a comprehensive solution for detecting fraudulent transactions within a synthetic financial dataset. The core of this solution is a custom-built, explainable machine learning model designed to handle the severe class imbalance typical of fraud detection scenarios. The model not only achieves high accuracy but also provides clear, human-readable explanations for its predictions.

## Dataset

The model is trained and evaluated on the "Synthetic Financial Datasets For Fraud Detection" from Kaggle. This dataset simulates mobile money transactions, featuring attributes like transaction type, amount, and account balances before and after the transaction for both the originator and the recipient. A binary label, `isFraud`, indicates the ground truth.

## Exploratory Data Analysis (EDA)

A thorough EDA was conducted to understand the dataset's characteristics and inform the feature engineering and modeling stages. Key findings include:

* **Severe Class Imbalance:** Fraudulent transactions constitute only about 0.13% of the dataset, a critical challenge that must be addressed in the modeling phase.

* **Fraudulent Transaction Types:** A crucial discovery was that fraudulent activities are confined exclusively to 'TRANSFER' and 'CASH_OUT' transaction types. This insight is directly incorporated as a business rule in the final model.

* **Data Distributions:** Numerical features like `amount` exhibited highly skewed distributions. Logarithmic transformations were applied during EDA to better visualize these patterns.

* **Temporal Patterns:** Analysis of the `step` feature (representing time) revealed that fraud rates fluctuate throughout the day, peaking in the early morning hours. This led to the creation of an `hour` feature to capture this cyclicality.

* **Feature Correlations:** A correlation heatmap highlighted expected relationships between financial features, confirming the dataset's internal consistency.

## Feature Engineering

To improve the model's predictive power, several new features were engineered based on EDA insights:

* **`hour`**: Extracted from the `step` feature (`step % 24`), this captures the time-of-day patterns associated with fraudulent activity.

* **`deltaOrig`**: Represents the discrepancy in the originator's account balance (`newbalanceOrig - oldbalanceOrg + amount`). In a legitimate transaction, this should be zero.

* **`deltaDest`**: Represents the discrepancy in the recipient's account balance (`newbalanceDest - oldbalanceDest - amount`). This helps identify inconsistencies in fund transfers.

## Modeling Strategy

The modeling approach is centered around a stacking ensemble method, which combines the predictions of multiple models to produce a more robust and accurate final prediction.

* **Base Models:** Two powerful and popular gradient boosting algorithms were chosen as base learners:
    * **LightGBM:** Known for its speed and efficiency, especially with large datasets.
    * **XGBoost:** Renowned for its performance and regularization capabilities.

* **Meta-Classifier:** A **Logistic Regression** model serves as the meta-classifier. It takes the probability outputs of the base models as input and learns the optimal way to combine them for the final fraud prediction.

This stacking architecture leverages the diverse strengths of both LightGBM and XGBoost, often leading to better performance than either model could achieve alone.

### Custom Model: `InsightAIMLModel`

To encapsulate the entire workflow from preprocessing to explainability, a custom Python class, `InsightAIMLModel`, was developed. This class provides a clean, reusable, and highly specialized solution for this fraud detection problem.

#### Key Features of `InsightAIMLModel`:

* **Integrated Preprocessing:** The model internally manages a preprocessing pipeline that handles one-hot encoding for categorical features like `type` and `hour`.

* **Stacking Ensemble Core:** The `fit` method orchestrates the training of the LightGBM and XGBoost base models, followed by the training of the logistic regression meta-model on their probability outputs.

* **Handling Class Imbalance:** The model directly addresses the severe class imbalance by using `class_weight='balanced'` for the LightGBM model and calculating `scale_pos_weight` for the XGBoost model. This ensures that the minority class (fraud) is given appropriate importance during training.

* **Intelligent Thresholding:** A critical feature is the custom threshold optimization. Instead of using a default 0.5 probability threshold, the model:
    1.  Splits the training data into a smaller training set and a validation set.
    2.  Trains the full stacked ensemble on the smaller training set.
    3.  Evaluates performance on the validation set across a range of probability thresholds (e.g., from 0.50 to 0.999).
    4.  It selects the optimal threshold that achieves a minimum recall of 99.5% while minimizing the number of false positives. This strategy is vital for a practical fraud detection system, where catching as much fraud as possible (high recall) is paramount, but minimizing disruption to legitimate customers (low false positives) is also crucial.

* **Business Rule Application:** Based on the EDA finding, the model incorporates a hard business rule in its `predict` method: a transaction can only be classified as fraudulent if its type is either 'TRANSFER' or 'CASH_OUT'. This significantly reduces the risk of false positives from other transaction types.

* **Built-in Explainability:** The model includes an `explain_text` method that uses the **SHAP (SHapley Additive exPlanations)** library. For any given transaction, this method provides:
    * A clear, final decision (FRAUD or NOT FRAUD).
    * A detailed summary of the model's reasoning, including the combined probability score versus the determined threshold.
    * A list of the top features that increased the risk of fraud (reasons).
    * A list of the top features that decreased the risk of fraud (counter-reasons), with their respective impacts.

## Evaluation Results

The final `InsightAIMLModel` was evaluated on a held-out test set, demonstrating exceptional performance.

**Performance Metrics:**

| Metric    | Score    |
| :-------- | :------- |
| Precision | 0.999593 |
| Recall    | 0.996347 |
| ROC-AUC   | 0.999773 |

**Confusion Matrix (Test Set):**

|                     | **Predicted Negative** | **Predicted Positive** |
| :------------------ | :--------------------- | :--------------------- |
| **Actual Negative** | 1,906,321              | 1                      |
| **Actual Positive** | 9                      | 2,455                  |

These results are outstanding. A **recall of 99.63%** means the model successfully identified 2,455 out of 2,464 fraudulent transactions. A **precision of 99.96%** is equally impressive, as it means that out of all transactions flagged as fraud, only a single one was a false positive. This balance is the hallmark of an effective and practical fraud detection system.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn lightgbm xgboost shap dill
    ```

2.  **Run the Notebook:** Open and execute the `InsightAI_Assignment.ipynb` notebook. This will automatically handle data download, analysis, model training, evaluation, and will serialize the trained `InsightAIMLModel` object to a file named `insightai_fraud_model.pkl`.

## File Descriptions

* `InsightAI_Assignment.ipynb`: The main Jupyter notebook containing the complete workflow.
* `model.py`: The Python script defining the `InsightAIMLModel` class.
* `README.md`: This detailed project overview.
