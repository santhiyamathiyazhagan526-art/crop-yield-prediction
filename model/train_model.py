import pandas as pd

# ======================================================
# STEP 1: Read Excel Dataset
# ======================================================

file_path = "../dataset/Crop_Yield_Dataset_Coimbatore_Erode_500_Records.xlsx"

coimbatore = pd.read_excel(file_path, sheet_name="Coimbatore")
erode = pd.read_excel(file_path, sheet_name="Erode")

# Combine both sheets into one dataset
data = pd.concat([coimbatore, erode], ignore_index=True)

print("====================================")
print("Dataset Loaded Successfully")
print("====================================")

print("\nDataset Shape:", data.shape)

print("\nFirst 5 Records")
print(data.head())


# ======================================================
# STEP 2: Select Input Features (X)
# ======================================================

X = data[
    [
        "Crop",
        "District",
        "Rainfall_mm",
        "Temperature_C",
        "Soil_pH",
        "Fertilizer",
    ]
]
# Keep a copy of the original input data
X_original = X.copy()

# ======================================================
# STEP 3: Select Target (y)
# ======================================================

y = data["Yield_ton_per_hectare"]

print("\nInput Features (X)")
print(X.head())

print("\nTarget (y)")
print(y.head())

# ======================================================
# STEP 4: Convert Text into Numbers
# ======================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_features = [
    "Crop",
    "District",
    "Fertilizer",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        )
    ],
    remainder="passthrough",
)

X_encoded = preprocessor.fit_transform(X)

print("\n====================================")
print("Encoding Completed Successfully")
print("====================================")

print("Original Shape :", X.shape)
print("Encoded Shape  :", X_encoded.shape)
# ======================================================
# STEP 5: Split Dataset into Training and Testing Data
# ======================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, X_train_original, X_test_original = train_test_split(
    X_encoded,
    y,
    X_original,
    test_size=0.20,
    random_state=42
)

print("\n====================================")
print("Dataset Split Completed")
print("====================================")

print("Training Data (X_train):", X_train.shape)
print("Testing Data  (X_test) :", X_test.shape)

print("Training Target (y_train):", y_train.shape)
print("Testing Target  (y_test) :", y_test.shape)
# ======================================================
# STEP 6: Train Linear Regression Model
# ======================================================

from sklearn.linear_model import LinearRegression

# Create the model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

print("\n====================================")
print("Linear Regression Model Trained Successfully")
print("====================================")
# ======================================================
# STEP 7: Make Predictions
# ======================================================

y_pred = linear_model.predict(X_test)

print("\nFirst 10 Predictions")
print(y_pred[:10])
# ======================================================
# STEP 8: Evaluate the Model
# ======================================================

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n====================================")
print("Linear Regression Performance")
print("====================================")

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R² Score : {r2:.4f}")
# ======================================================
# STEP 9: Train Decision Tree Regression Model
# ======================================================

from sklearn.tree import DecisionTreeRegressor

# Create the model
decision_tree = DecisionTreeRegressor(
    random_state=42
)

# Train the model
decision_tree.fit(X_train, y_train)

# Predict
dt_predictions = decision_tree.predict(X_test)

print("\n====================================")
print("Decision Tree Model Trained Successfully")
print("====================================")
# ======================================================
# STEP 10: Evaluate Decision Tree
# ======================================================

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_predictions)

print("\nDecision Tree Performance")

print(f"MAE      : {dt_mae:.4f}")
print(f"MSE      : {dt_mse:.4f}")
print(f"RMSE     : {dt_rmse:.4f}")
print(f"R² Score : {dt_r2:.4f}")
# ======================================================
# STEP 11: Train Random Forest Regression
# ======================================================

from sklearn.ensemble import RandomForestRegressor

# Create the model
random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train the model
random_forest.fit(X_train, y_train)

# Predict
rf_predictions = random_forest.predict(X_test)

print("\n====================================")
print("Random Forest Model Trained Successfully")
print("====================================")
# ======================================================
# STEP 12: Evaluate Random Forest
# ======================================================

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Performance")

print(f"MAE      : {rf_mae:.4f}")
print(f"MSE      : {rf_mse:.4f}")
print(f"RMSE     : {rf_rmse:.4f}")
print(f"R² Score : {rf_r2:.4f}")
# ======================================================
# STEP 13: Compare All Models
# ======================================================

import pandas as pd

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "MAE": [
        mae,
        dt_mae,
        rf_mae
    ],
    "MSE": [
        mse,
        dt_mse,
        rf_mse
    ],
    "RMSE": [
        rmse,
        dt_rmse,
        rf_rmse
    ],
    "R2 Score": [
        r2,
        dt_r2,
        rf_r2
    ]
})

print("\n====================================")
print("MODEL COMPARISON")
print("====================================")
print(results)

best_model = results.loc[results["R2 Score"].idxmax()]

print("\n====================================")
print("BEST MODEL")
print("====================================")
print(best_model)
# ======================================================
# STEP 14: Save the Random Forest Model
# ======================================================

import joblib
import os

# Create folder if it doesn't exist
os.makedirs("../saved_model", exist_ok=True)

# Save the trained model
joblib.dump(linear_model, "../saved_model/linear_regression_model.pkl")
# Save the encoder (preprocessor)
joblib.dump(preprocessor, "../saved_model/preprocessor.pkl")

print("\n====================================")
print("Model Saved Successfully")
print("====================================")
print("Model File : linear_regression_model.pkl")
print("Preprocessor : preprocessor.pkl")

# ======================================================
# STEP 15: Display Prediction Results
# ======================================================

comparison = X_test_original[["District", "Crop"]].copy()

comparison["Actual Yield"] = y_test.values
comparison["Predicted Yield"] = np.round(y_pred, 2)

print("\n====================================")
print("ACTUAL VS PREDICTED RESULTS")
print("====================================")
print(comparison.head(10))

# Save all prediction results to Excel
comparison.to_excel(
    "../saved_model/prediction_results.xlsx",
    index=False
)

print("\nPrediction results saved as: prediction_results.xlsx")
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Actual Yield vs Predicted Yield")

plt.grid(True)

plt.show()