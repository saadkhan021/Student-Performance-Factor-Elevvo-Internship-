# Student-Performance-Factor-Elevvo-Internship-
# Student Exam Score Prediction

This project predicts student exam scores using machine learning techniques based on various student performance factors such as teacher quality, parental education level, and distance from home.

The goal is to compare **Linear Regression**, **Polynomial Regression (with Ridge regularization)**, and **Random Forest Regression** to find the most accurate model for predicting exam scores.

---

## Dataset

- **File:** `StudentPerformanceFactors.csv`
- **Features include:**
  - `Teacher_Quality`
  - `Parental_Education_Level`
  - `Distance_from_Home`
  - Other numeric and categorical student performance factors
- **Target:** `Exam_Score`

**Missing Values Handling:**
- Categorical features are filled with mode (most frequent value)
- Numeric features are scaled using StandardScaler

---

## Project Steps

1. **Data Preprocessing**
   - Handle missing values
   - Separate numerical and categorical features
   - Encode categorical features using `OrdinalEncoder`
   - Scale numerical features using `StandardScaler`

2. **Train-Test Split**
   - Split the dataset into 80% training and 20% testing

3. **Modeling**
   - **Linear Regression**: baseline regression model
   - **Polynomial Regression (Ridge)**: polynomial features with degree 2 or 3 and Ridge regularization; hyperparameters tuned using `GridSearchCV`
   - **Random Forest Regression**: tree-based ensemble model to capture non-linear patterns

4. **Evaluation Metrics**
   - **R² Score**: proportion of variance explained by the model
   - **MAE (Mean Absolute Error)**: average absolute difference between actual and predicted values
   - **MSE (Mean Squared Error)**: average squared difference between actual and predicted values

5. **Visualization**
   - Scatter plots of actual vs predicted scores for Linear and Polynomial Regression
   - Feature importance plot from Random Forest model

---

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
