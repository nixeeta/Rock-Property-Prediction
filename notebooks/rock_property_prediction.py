import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("../data/rock_data.csv")  # Change to your actual file path

# ------------------------ 🧹 Data Preprocessing ------------------------

# Example: Predicting "Compressive Strength"
target_col = "CompressiveStrength"  # change this to your actual target column name
print("Columns in dataset:", data.columns.tolist())
X = data.drop(columns=[target_col])
y = data[target_col]

# Optional: Handle missing values
X = X.fillna(X.mean())

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------ 🧠 Train Models ------------------------

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# XGBoost Regressor
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# ------------------------ 📊 Evaluation ------------------------

def evaluate_model(y_true, y_pred, name):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"🔹 {name} Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print("-" * 40)
    return {"Model": name, "R²": r2, "MAE": mae, "RMSE": rmse}

results = []
results.append(evaluate_model(y_test, y_pred_lin, "Linear Regression"))
results.append(evaluate_model(y_test, y_pred_dt, "Decision Tree"))
results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))

# ------------------------ 📈 Plot Results ------------------------

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot R² comparison
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="R²", data=results_df, palette="coolwarm")
plt.title("Model R² Score Comparison")
plt.ylabel("R² Score")
plt.ylim(0, 1)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()