# Bank Marketing Subscription Prediction

## Using Machine Learning to Improve Term-Deposit Campaign Targeting

A machine learning classification project that predicts which bank customers are more likely to subscribe to a term deposit, helping marketing teams prioritize customers for telephone campaigns.

The project focuses on building a **deployable pre-call prediction model** using only information available before a marketing call begins.

---

## Project Overview

Bank marketing campaigns often involve contacting large numbers of customers by telephone. Calling every customer can require significant time and resources.

The objective of this project was to develop a machine learning model that can estimate the probability that a customer will subscribe to a term deposit and support more targeted campaign prioritization.

The project follows a complete machine learning workflow:

**Business Understanding → Data Understanding → Data Quality → Exploratory Data Analysis → Leakage Investigation → Preprocessing → Baseline → Model Development → Evaluation → Interpretation → Responsible Deployment**

---

## Business Question

> **Can we use customer and campaign information available before a telephone call to identify customers who are more likely to subscribe to a term deposit?**

The model is intended to support **customer prioritization**, not to make automatic decisions about customers.

---

## Dataset

The project uses the **Bank Marketing dataset**, containing information about customers and previous marketing campaign interactions.

- **Records:** 45,211
- **Features:** 16 input variables
- **Target:** Term-deposit subscription
- **Target values:** `yes` / `no`
- **Classification type:** Binary classification

The original target was encoded as:

- `no` → `0`
- `yes` → `1`

### Key Features

Examples of variables used include:

- Age
- Job
- Marital status
- Education
- Account balance
- Housing loan
- Personal loan
- Contact method
- Campaign month
- Number of campaign contacts
- Previous campaign outcome

---

## Prediction Moment

A key design decision in this project was defining **when the prediction is supposed to be made**.

The prediction moment was defined as:

> **Immediately before the telephone call begins.**

This means the model should only use information that would realistically be available at that point.

---

## Data Leakage Investigation

The variable `duration` represents the length of the telephone call.

Because call duration is only known during or after the call, using it to predict whether the customer will subscribe would introduce **data leakage**.

To demonstrate the impact of leakage, I conducted a controlled comparison.

| Model | ROC-AUC |
|---|---:|
| Deployment model — without `duration` | **0.7717** |
| Diagnostic model — with `duration` | **0.9056** |

Although the model containing `duration` achieved a higher ROC-AUC, it was not suitable for deployment because the information would not be available at the prediction moment.

Therefore:

> **`duration` was excluded from the final deployment model.**

This demonstrates an important machine learning principle:

> **The most predictive model is not necessarily the most deployable model.**

---

## Data Preparation

The dataset was examined for:

- Data types
- Missing values
- Duplicate records
- Unique values
- Categorical levels
- Potentially suspicious values
- Class imbalance
- Semantic `unknown` values

Values such as `unknown` were not automatically removed because an unknown value may represent missing information in the original data collection process rather than an invalid observation.

---

## Exploratory Data Analysis

Exploratory analysis was performed to understand the structure of the dataset and identify associations between customer/campaign characteristics and subscription outcomes.

Subscription rates were examined across variables including:

- Job
- Education
- Contact method
- Campaign month
- Previous campaign outcome
- Housing loan status

The target variable was also examined to understand the class imbalance between subscribers and non-subscribers.

---

## Machine Learning Workflow

### 1. Train/Test Split

The data was divided into:

- **80% training data**
- **20% test data**

A stratified split was used to preserve the target-class distribution.

### 2. Preprocessing

Numerical and categorical variables were handled separately.

**Numerical features**
- Standardized for Logistic Regression

**Categorical features**
- One-hot encoded

The preprocessing was implemented using a Scikit-learn pipeline so that transformations were fitted using the training data and applied consistently to test and future data.

---

## Models Evaluated

Three approaches were evaluated:

### Dummy Classifier

Used as a baseline to establish the performance level expected from a trivial classifier.

### Logistic Regression

Selected as the primary classification model because it is appropriate for binary classification, relatively interpretable and provides probability estimates useful for customer prioritization.

### Decision Tree

Used as a comparison model capable of representing non-linear relationships and interactions.

---

## Model Performance

The final deployment models were evaluated on the held-out test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | **89.33%** | **66.32%** | **17.86%** | **0.28** | **0.7717** |
| Decision Tree | 89.17% | 65.61% | 15.69% | 0.25 | 0.7056 |
| Dummy Baseline | — | — | — | — | 0.5000 |

### Selected Model

**Logistic Regression**

The Logistic Regression model achieved the highest ROC-AUC among the valid deployment models and substantially outperformed the dummy baseline.

---

## Understanding the Metrics

Because the dataset is imbalanced, accuracy was not used as the only evaluation metric.

### ROC-AUC — 0.7717

Measures the model's ability to distinguish between customers who subscribed and those who did not across different classification thresholds.

### Precision — 66.32%

Among customers predicted as likely subscribers at the 0.50 threshold, approximately 66% were actual subscribers.

### Recall — 17.86%

At the 0.50 threshold, the model identified approximately 18% of the customers who actually subscribed.

### F1-Score — 0.28

Provides a combined measure of precision and recall.

---

## Feature Importance

Permutation importance was used to understand which variables the model relied on most for prediction.

The five strongest features were:

1. `contact`
2. `month`
3. `poutcome`
4. `campaign`
5. `housing`

Permutation importance measures the reduction in model performance when a feature is randomly shuffled.

These results represent **predictive associations**, not causal relationships.

For example, the importance of `contact` does not mean that changing the contact method will necessarily cause a customer to subscribe.

---

## Threshold Analysis

The default classification threshold of 0.50 is not necessarily the optimal business threshold.

Changing the threshold creates a trade-off:

**Lower threshold**
- More customers are prioritized
- Recall generally increases
- More calls may be required
- Precision may decrease

**Higher threshold**
- Fewer customers are prioritized
- Precision may increase
- Fewer calls may be required
- More potential subscribers may be missed

Therefore, the final threshold should be selected according to:

- Available calling capacity
- Cost of contacting customers
- Cost of missing potential subscribers
- Desired campaign coverage

---

## Responsible Model Use

The model should be used as a **prioritization tool**, rather than an automatic customer-selection or exclusion system.

Recommended controls include:

- Human oversight of campaign decisions
- Respect for customer consent and opt-out requirements
- Monitoring model performance over time
- Monitoring subgroup performance
- Monitoring changes in customer and campaign behavior
- Reviewing data quality and feature availability
- Retraining when model performance deteriorates

---

## Model Artifact

The trained model was saved as:

`bank_marketing_model.joblib`

The saved artifact can be used to generate predictions on new customer records using the same trained preprocessing and modeling workflow.

---

## Project Files

| File | Description |
|---|---|
| `Bank_Marketing_Project.ipynb` | Complete Jupyter Notebook containing the analysis and machine learning workflow |
| `bank-full.csv` | Bank Marketing dataset |
| `bank_marketing_model.joblib` | Saved trained deployment model |
| `tree_plot.svg` | Visualization of the Decision Tree model |

---

## Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- GitHub

---

## Key Insights

1. **The prediction moment matters.**  
   Including post-call information such as `duration` increased ROC-AUC from 0.7717 to 0.9056, but introduced data leakage and was therefore excluded.

2. **Accuracy alone can be misleading.**  
   Although the Logistic Regression model achieved 89.33% accuracy, its recall at the 0.50 threshold was 17.86%, demonstrating the importance of evaluating multiple metrics in an imbalanced classification problem.

3. **Customer prioritization requires a threshold decision.**  
   The probability threshold determines the balance between identifying more potential subscribers and limiting the number of customers prioritized for calls.

4. **Model interpretation matters.**  
   Contact method, month, previous campaign outcome, campaign contacts and housing status were among the features most relied upon by the model.

---

## Limitations

- The dataset represents historical marketing activity from a specific banking context.
- Customer behavior and campaign effectiveness may change over time.
- `unknown` categories may reflect the original data collection process.
- Predictive relationships should not be interpreted as causal effects.
- The project does not include complete campaign cost or customer lifetime-value information.
- Subgroup performance may vary and should be monitored after deployment.

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer subscription prediction.

The final model achieved a test ROC-AUC of **0.7717** while deliberately excluding post-call information that would create data leakage.

Rather than automatically deciding which customers should or should not be contacted, the model can provide probability scores that support marketing teams in prioritizing customers according to campaign capacity and business objectives.

---

## Author

**Nkechika Julian Nnodiogu**

Data Analyst | Data Scientist | Medical Laboratory Scientist

[LinkedIn](https://www.linkedin.com/in/nkechika-julian-nnodiogu-b7714918b/)
