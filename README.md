# 🚀 Contextual Predictive Maintenance (IoT Edge AI)

> **Advanced Data Science & Machine Learning Project**
>
> Predicting machine failures using Industrial IoT sensor data and Machine Learning.

---

# 📖 Project Overview

Predictive Maintenance is a proactive maintenance strategy that leverages machine learning to predict equipment failures before they occur. Instead of relying on scheduled maintenance or reacting to breakdowns, predictive maintenance enables industries to optimize maintenance schedules, reduce downtime, and improve operational efficiency.

This project focuses on building a machine learning pipeline that analyzes industrial IoT sensor data to predict machine failures. The solution combines data preprocessing, feature engineering, exploratory data analysis, and multiple machine learning models to identify failure patterns and provide reliable predictions.

---

# 🎯 Objectives

- Develop an intelligent machine failure prediction system.
- Analyze Industrial IoT sensor data.
- Perform feature engineering to improve model performance.
- Compare multiple machine learning algorithms.
- Build a scalable predictive maintenance pipeline.
- Provide insights that help reduce downtime and maintenance costs.

---

# 🏭 Business Problem

Unexpected equipment failures can lead to:

- Increased maintenance costs
- Production downtime
- Reduced operational efficiency
- Financial losses
- Safety risks

The goal of this project is to predict failures in advance, enabling preventive maintenance and improving overall equipment reliability.

---

# 📂 Dataset Information

### Dataset

**AI4I 2020 Predictive Maintenance Dataset**

**Source**

https://www.kaggle.com/datasets/shivamb/ai4i2020-predictive-maintenance-dataset

---

## Dataset Description

The dataset simulates real-world industrial machine operations using multiple IoT sensor readings. Each record represents the operational state of a manufacturing machine along with its failure status.

### Features

| Feature | Description |
|----------|-------------|
| UDI | Unique Identifier |
| Product ID | Machine Product Identifier |
| Type | Machine Type (L, M, H) |
| Air Temperature [K] | Ambient Air Temperature |
| Process Temperature [K] | Internal Machine Temperature |
| Rotational Speed [rpm] | Machine Rotational Speed |
| Torque [Nm] | Torque Produced |
| Tool Wear [min] | Tool Usage Duration |
| Machine Failure | Target Variable |

---

## Target Variable

**Machine Failure**

- **0** → No Failure
- **1** → Machine Failure

This is a **Binary Classification** problem.

---

# 🛠 Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Kaggle Notebooks
- Git & GitHub

---

# 📁 Project Structure

```
predictive-maintenance-ai/
│
├── notebooks/
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── modeling/
│   └── utils/
│
├── reports/
├── images/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Exploratory Data Analysis (EDA)

The dataset was analyzed to understand its structure and quality before model development.

EDA included:

- Dataset overview
- Shape and dimensionality
- Data type analysis
- Missing value detection
- Duplicate record analysis
- Statistical summary
- Target variable distribution
- Sensor feature visualization
- Correlation analysis

These analyses provided insights into machine behavior and prepared the dataset for feature engineering.

---

# ⚙️ Feature Engineering

Feature engineering was performed to extract meaningful information from raw sensor readings and improve predictive performance.

### Rolling Mean

Rolling averages were computed for important sensor readings to capture recent operational trends rather than relying on individual observations.

**Purpose**

- Smooth noisy sensor readings
- Capture recent machine behavior
- Identify gradual operational changes

---

### Rolling Standard Deviation

Rolling standard deviation was calculated to measure sensor stability over time.

**Purpose**

- Detect abnormal fluctuations
- Identify unstable operating conditions
- Improve anomaly detection

---

### Rolling Variance

Rolling variance was generated to quantify variability within sensor readings.

**Purpose**

- Measure consistency
- Detect irregular machine behavior
- Capture operational instability

---

### Why Feature Engineering?

Raw sensor values often fail to represent machine behavior over time.

Feature engineering enables the model to learn:

- Operational trends
- Sensor stability
- Machine variability
- Hidden failure patterns

These engineered features provide richer information for predictive modeling and improve model performance.

---

# 🤖 Machine Learning Models

To establish strong baseline performance, multiple supervised classification algorithms were implemented and compared.

### Logistic Regression

- Linear baseline classifier
- Fast and interpretable
- Simple benchmark model

---

### Decision Tree

- Rule-based classification
- Handles nonlinear relationships
- Easy to interpret

---

### Random Forest

- Ensemble learning algorithm
- Reduces overfitting
- Improves prediction stability

---

### Support Vector Machine (SVM)

- Margin-based classifier
- Effective for complex classification problems
- Strong performance on structured datasets

---

### XGBoost

- Gradient Boosting algorithm
- High predictive accuracy
- Captures complex feature interactions
- Widely used in industrial machine learning applications

---

# 📈 Model Evaluation

Models are evaluated using standard classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

Performance comparison helps identify the most effective algorithm for predictive maintenance.

---

# 🔮 Future Enhancements

The project can be further enhanced by incorporating:

- Contextual weather information
- Factory load conditions
- Advanced feature engineering
- SMOTE for handling class imbalance
- LightGBM implementation
- Hyperparameter optimization
- Cross-validation
- SHAP Explainability
- Precision-Recall optimization
- Real-time prediction pipeline
- Interactive Streamlit Dashboard

---

# 📈 Expected Outcomes

The completed system aims to:

- Predict machine failures before occurrence
- Reduce maintenance costs
- Minimize unexpected downtime
- Improve production efficiency
- Support proactive maintenance decisions

---

# 📌 Repository Workflow

The project follows industry-standard software engineering practices:

- Modular code organization
- Version control using Git & GitHub
- Semantic commit messages
- Incremental development
- Clean project documentation

---

# 📚 References

- AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- Scikit-learn Documentation
- XGBoost Documentation
- Pandas Documentation
- NumPy Documentation

---

# 👨‍💻 Author

**Rohan Thakar**

Advanced Data Science & Machine Learning Intern

Infotact Solutions

---

# ⭐ Project Status

**In Progress**

This project is actively being developed with continuous improvements in feature engineering, model optimization, and predictive performance.
