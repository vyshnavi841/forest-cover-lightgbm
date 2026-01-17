# Forest Cover Type Classification

## Project Overview
This project implements an end-to-end machine learning pipeline to predict forest cover types using cartographic and environmental features. The task is a **multi-class classification problem** with seven forest cover classes.

The project focuses on building robust models, handling class imbalance, improving performance through feature engineering, tuning models systematically, and comparing different ensemble learning approaches.

---

## Objective
The goal of this project is to:
- Predict the forest cover type for a given land area
- Handle large-scale, imbalanced data
- Improve predictive performance using feature engineering and hyperparameter tuning
- Compare model performance and analyze trade-offs between accuracy and efficiency

---

## Dataset
- **Source:** UCI Machine Learning Repository – Forest Cover Type Dataset
- **Number of Samples:** ~580,000
- **Target Variable:** `Cover_Type` (7 classes)
- **Features:** Elevation, slope, distances to hydrology and roads, hillshade indices, wilderness areas, soil types
- **Missing Values:** None

Each data point represents a 30m × 30m area of forest land.

---

## Project Structure
```text
forest-cover-lightgbm/
├── data/
│   └── forest_cover.csv
├── notebook/
│   └── forest_cover_classification.ipynb
├── results/
├── requirements.txt
└── README.md

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Examined dataset shape, data types, and feature distributions
- Analyzed class imbalance in the target variable
- Studied correlations to guide feature engineering

---

### 2. Feature Engineering
To improve model performance, multiple new features were created, including:
- Interaction features (e.g., elevation × slope)
- Aggregated hillshade features
- Distance-based transformations
- Log-scaled distance features
- Binned elevation features

These features help capture non-linear relationships present in geographical data.

---

### 3. Data Preparation
- Converted relevant features to categorical data types
- Used stratified train–test split to preserve class distribution
- No one-hot encoding was required due to tree-based models

---

### 4. Baseline Model
- A baseline **LightGBM** classifier was trained using default parameters
- Class imbalance handled using balanced class weights
- Baseline metrics recorded for comparison

---

### 5. Hyperparameter Tuning
- Hyperparameter optimization performed using **Optuna**
- Used **Stratified K-Fold Cross-Validation**
- Optimized for **macro F1-score** to ensure balanced class performance
- Tuned parameters included:
  - `num_leaves`
  - `learning_rate`
  - `feature_fraction`
  - `reg_alpha`
  - `reg_lambda`

---

### 6. Final Model Training & Evaluation
The tuned LightGBM model was evaluated using:
- Macro and Micro Precision
- Macro and Micro Recall
- Macro and Micro F1-score
- Confusion Matrix
- Feature Importance (split-based and gain-based)

---

### 7. Comparative Analysis
A **Random Forest** classifier was trained as an alternative ensemble model and compared with the tuned LightGBM model.

#### Key Findings:
- **Random Forest achieved higher macro and micro F1-scores**, indicating stronger predictive performance on this dataset.
- **LightGBM trained faster and scaled more efficiently**, making it suitable for large datasets and time-constrained environments.

This comparison highlights the trade-off between predictive performance and computational efficiency.

---

## Results Summary
- Random Forest achieved the best F1-scores in this experiment
- LightGBM performed competitively with significantly faster training time
- Feature engineering and hyperparameter tuning substantially improved performance over baseline models

---

## Tools & Technologies
- **Programming Language:** Python
- **Libraries:** LightGBM, Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Hyperparameter Tuning:** Optuna
- **Environment:** Jupyter Notebook

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```
### 2. Install Dependencies
pip install -r requirements.txt


### 3. Run the Notebook
jupyter notebook

### Open and run : 
notebook/forest_cover_classification.ipynb

## Reproducibility Notes
- Results may vary slightly due to randomness in model training
- Hyperparameter tuning outcomes depend on the number of trials
- The notebook is fully self-contained and reproducible

## Conclusion
This project demonstrates a complete machine learning workflow for multi-class classification on a large real-world dataset. While Random Forest achieved higher predictive performance in this experiment, LightGBM offered superior training efficiency and scalability.

The project emphasizes that effective model selection depends on both performance metrics and practical deployment considerations.

### Author
Vyshnavi Kaki
