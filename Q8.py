import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns

rng = np.random.default_rng(17010868)

# === Load Datasets ===
# File paths
num_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneNum.csv"

# Define column names
num_columns = [
    "Average Rating", "Average Difficulty", "Number of Ratings",
    "Received a 'pepper'?", "Proportion Would Retake",
    "Online Ratings Count", "Male", "Female"
]

df_num = pd.read_csv(num_file, header=None, names=num_columns)

df_num.head()

tags_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneTags.csv"

tags_columns = [
    "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
    "Participation Matters", "Don’t Skip Class", "Lots of Homework",
    "Inspirational", "Pop Quizzes", "Accessible", "So Many Papers",
    "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
    "Amazing Lectures", "Caring", "Extra Credit", "Group Projects",
    "Lecture Heavy"
]
df_tags = pd.read_csv(tags_file, header=None, names=tags_columns)

df_tags.head()

qual_file = "/Users/kangkyeongmo/Desktop/School/2024_Fall/DS-GA_1001_001/Group_Project/Capstone/rmpCapstoneQual.csv"

qual_columns = ["Major/Field", "University", "State"]

df_qual = pd.read_csv(qual_file, header=None, names=qual_columns)

df_qual.head

filtered_df_num = df_num.dropna(subset=['Number of Ratings'])
print(np.mean(filtered_df_num['Number of Ratings']))
filtered_df_num = filtered_df_num[filtered_df_num['Number of Ratings'] > np.mean(filtered_df_num['Number of Ratings'])]
filtered_df_tags = df_tags.loc[filtered_df_num.index]
filtered_df_qual = df_qual.loc[filtered_df_num.index]

# Normalize Tag Data 
filtered_df_tags_normalized = filtered_df_tags.div(filtered_df_num["Number of Ratings"], axis=0)
print(filtered_df_tags_normalized.head())

# === Prepare Data ===
# Predictors: Normalized Tags
X = filtered_df_tags_normalized

# Target: Average Rating
y = filtered_df_num["Average Rating"]

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17010868) ### N_Number

# === Standardize Predictors ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)       # Transform test data

# === Fit Regression Model ===
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Linear Regression - R^2: {r2:.4f}")
print(f"Linear Regression - RMSE: {rmse:.4f}")

# === Analyze Coefficients ===
coefficients = pd.DataFrame({
    "Tag": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\nRegression Coefficients:")
print(coefficients)

# === Visualize Coefficients ===
plt.figure(figsize=(10, 6))
plt.bar(coefficients["Tag"], coefficients["Coefficient"])
plt.xticks(rotation=45)
plt.title("Regression Coefficients for Each Tag")
plt.xlabel("Tags")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.show()

### === Address Multicollinearity: Calculate VIF ===
# Rules of Thumb:
    # VIF = 1 : No correlation with other features.
	# VIF > 5 : Moderate correlation (needs attention).
	# VIF > 10 : Severe multicollinearity (should likely be addressed).
# Add a constant(intercept) for VIF calculation
X_with_const = np.hstack([np.ones((X_train.shape[0], 1)), X_train.values])
vif_data = pd.DataFrame({
    "Feature": ["Intercept"] + list(X.columns),
    "VIF": [variance_inflation_factor(X_with_const, i) for i in range(X_with_const.shape[1])]
})

print("\nVariance Inflation Factors (VIF):")
print(vif_data)

# Highlight features with high VIF
high_vif = vif_data[vif_data["VIF"] > 5]
if not high_vif.empty:
    print("\nFeatures with High Multicollinearity (VIF > 5):")
    print(high_vif)

# Compute the correlation matrix
corr_matrix = X_train.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Optional: Adjust the figure size for better readability
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

# Display the plot
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

