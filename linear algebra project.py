

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Set plot style for better visualization
sns.set(style="whitegrid")


# ============================================================
# STEP 1: Load Dataset from Local System
# ============================================================

# Always make sure the CSV file exists in the working directory
# You can also provide the full absolute path if needed

CSV_FILE = "bank.csv"

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"{CSV_FILE} not found in working directory")

# Read the dataset
df = pd.read_csv(CSV_FILE, encoding="utf-8")

print("Dataset loaded successfully!\n")


# ============================================================
# STEP 2: Inspect Dataset Structure
# ============================================================

# Display first few rows
print("First 5 rows of dataset:")
print(df.head(), "\n")

# Display dataset information (columns, datatypes, memory)
print("Dataset Info:")
print(df.info(), "\n")

# Display column names
print("Column Names:")
print(df.columns.tolist(), "\n")


# ============================================================
# STEP 3: Check and Handle Missing Values
# ============================================================

# Check missing values per column
print("Missing values per column:")
print(df.isnull().sum(), "\n")

# Drop rows containing any missing values
df.dropna(inplace=True)

print("Dataset shape after dropping NaN values:", df.shape, "\n")


# ============================================================
# STEP 4: Exploratory Data Visualization
# ============================================================

# Scatter Plot: Age vs Balance
plt.figure(figsize=(8, 5))
plt.scatter(df["age"], df["balance"], alpha=0.5)
plt.xlabel("Age")
plt.ylabel("Balance")
plt.title("Scatter Plot: Age vs Balance")
plt.show()

# Bar Plot: Job Distribution
plt.figure(figsize=(10, 5))
sns.countplot(data=df, x="job")
plt.xticks(rotation=45)
plt.title("Bar Plot: Job Distribution")
plt.show()

# Box Plot: Balance (for outlier detection)
plt.figure(figsize=(6, 4))
sns.boxplot(x=df["balance"])
plt.title("Box Plot: Balance")
plt.show()


# ============================================================
# STEP 5: Outlier Detection and Removal (Using IQR Method)
# ============================================================

Q1 = df["balance"].quantile(0.25)
Q3 = df["balance"].quantile(0.75)
IQR = Q3 - Q1

# Keep only non-outlier data
df = df[
    (df["balance"] >= Q1 - 1.5 * IQR) &
    (df["balance"] <= Q3 + 1.5 * IQR)
]

print("Dataset shape after removing outliers:", df.shape, "\n")


# ============================================================
# STEP 6: Encode Target Variable
# ============================================================

# Convert target column 'deposit' from text to numeric
# yes -> 1, no -> 0
df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

print("Target variable encoded successfully\n")


# ============================================================
# STEP 7: Encode Categorical Features
# ============================================================

# Convert categorical columns into dummy/one-hot encoded variables
df = pd.get_dummies(df, drop_first=True)

print("Categorical features encoded\n")


# ============================================================
# STEP 8: Split Features and Target
# ============================================================

X = df.drop("deposit", axis=1)  # Input features
y = df["deposit"]               # Target label

print("Features and target separated\n")


# ============================================================
# STEP 9: Feature Scaling
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed\n")


# ============================================================
# STEP 10: Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train-test split completed\n")


# ============================================================
# STEP 11: Model Selection and Training
# ============================================================

# Logistic Regression is chosen because:
# - Target variable is binary (yes/no)
# - Model is interpretable
# - Works well for tabular data

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed\n")


# ============================================================
# STEP 12: Model Testing / Prediction
# ============================================================

y_pred = model.predict(X_test)

print("Model prediction completed\n")


# ============================================================
# STEP 13: Prediction vs Actual Visualization
# ============================================================

plt.figure(figsize=(10, 4))
plt.scatter(range(len(y_test)), y_test, label="Actual", alpha=0.7)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted", alpha=0.5)
plt.xlabel("Sample Index")
plt.ylabel("Target Value")
plt.title("Actual vs Predicted Values")
plt.legend()
plt.show()


# ============================================================
# STEP 14: Model Evaluation Metrics
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:")
print(accuracy, "\n")

print("Confusion Matrix:")
print(conf_matrix, "\n")

print("Classification Report:")
print(class_report)


# ============================================================
# END OF PROJECT
# ============================================================
