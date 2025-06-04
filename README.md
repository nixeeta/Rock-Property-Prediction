# 🧪 Machine Learning Project: Rock Property Prediction

## 📍 Objective
To build a predictive model that estimates the physical or mechanical properties of rocks (such as porosity, permeability, or compressive strength) using geological and geophysical input features.

## 📂 Dataset
- **Source**: https://www.kaggle.com/datasets/s3programmerlead/rock-strength-prediction-dataset
- **Features**:
  - [Feature1] — e.g., Depth
  - [Feature2] — e.g., Density
  - [Feature3] — e.g., Sonic Velocity
  - [Feature4] — e.g., Resistivity
- **Target Variable**:
  - [e.g., Porosity / Compressive Strength / Permeability]

## 🔍 Steps Followed

### 1. 🧼 Data Cleaning & Preprocessing
- Removed missing values and outliers
- Normalized or scaled features using StandardScaler/MinMaxScaler
- Performed train-test split (80:20)

### 2. 🏗️ Feature Engineering
- Correlation analysis to select relevant features
- Added polynomial features / interaction terms (if applicable)

### 3. 🤖 Model Training
Trained the following models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

### 4. 📊 Evaluation
Used the following metrics to evaluate performance:
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

| Model                 | R² Score | MAE   | RMSE  |
|----------------------|----------|-------|-------|
| Linear Regression     | 0.5758     | 0.5332  | 0.7456  |
| Random Forest Regressor | 0.8051     | 0.3275  | 0.5053  |
| XGBoost Regressor     | 0.8301     | 0.3096  | 0.4718  |

### 5. 🛠️ Hyperparameter Tuning
Used GridSearchCV / RandomizedSearchCV to optimize:
- `n_estimators`
- `max_depth`
- `learning_rate` (for XGBoost)

### 6. ✅ Final Model Selection
- Selected the model with best validation performance
- Saved the model using `joblib` or `pickle`

## 📈 Visualizations
- Correlation heatmap of input features
- Actual vs Predicted plot for the best model
- Feature importance chart (for tree-based models)
