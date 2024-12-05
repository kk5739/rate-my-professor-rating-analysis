import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# === Load Datasets ===
# File paths
num_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneNum.csv"
tags_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneTags.csv"
qual_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneQual.csv"

# Define column names
num_columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a 'pepper'?", "Proportion Would Retake",
    "Online Ratings Count", "Male", "Female"
]
tags_columns = [
    "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
    "Participation Matters", "Don’t Skip Class", "Lots of Homework",
    "Inspirational", "Pop Quizzes", "Accessible", "So Many Papers",
    "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects",
    "Lecture Heavy"
]
qual_columns = ["Major/Field", "University", "State"]

# Load datasets
df_num = pd.read_csv(num_file, header=None, names=num_columns)
df_tags = pd.read_csv(tags_file, header=None, names=tags_columns)
df_qual = pd.read_csv(qual_file, header=None, names=qual_columns)

# Prepare data
X = df_tags
y = df_num["Average Rating"]

# Handle missing values
X = X.fillna(0)
y = y.fillna(y.mean())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R^2: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Identify the most predictive tag
coefficients = pd.DataFrame({
    "Tag": tags_columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nMost Predictive Tags:")
print(coefficients.head(3))

# Address multicollinearity: Calculate VIF
X_with_constant = np.hstack([np.ones((X_train.shape[0], 1)), X_train])  # Add constant for VIF calculation
vif_data = pd.DataFrame({
    "Feature": ["Intercept"] + tags_columns,
    "VIF": [variance_inflation_factor(X_with_constant, i) for i in range(X_with_constant.shape[1])]
})

print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# Highlight features with high VIF
high_vif = vif_data[vif_data["VIF"] > 5]
if not high_vif.empty:
    print("\nFeatures with High Multicollinearity (VIF > 5):")
    print(high_vif)
