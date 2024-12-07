import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)


rng = np.random.default_rng(17010868) 
rng = rng.integers(1,1000000)

column_names_numerical = ["Average Rating", 
            "Average Difficulty", 
            "Number of ratings",
            "Received a 'pepper'?",
            "The proportion of students that said they would take the class again",
            "The number of ratings coming from online classes",
            "Male",
            "Female"]

column_names_tags = ["Tough grader", 
                 "Good feedback", 
                 "Respected", 
                 "Lots to read", 
                 "Participation matters", 
                 "Don’t skip class or you will not pass", 
                 "Lots of homework", 
                 "Inspirational", 
                 "Pop quizzes!", 
                 "Accessible", 
                 "So many papers", 
                 "Clear grading", 
                 "Hilarious",
                 "Test heavy", 
                 "Graded by few things", 
                 "Amazing lectures", 
                 "Caring", 
                 "Extra credit", 
                 "Group projects", 
                 "Lecture heavy"]

# Get Numerical factors
df_numerical = pd.read_csv('rmpCapstoneNum.csv', header=None, names = column_names_numerical)
# Get Tags factors
df_tags = pd.read_csv('rmpCapstoneTags.csv', header = None, names = column_names_tags)

df_numerical = df_numerical.reset_index(drop=True)
df_tags = df_tags.reset_index(drop=True)

# Combine them into one df
df = pd.concat([df_numerical, df_tags], axis=1)
print(df)

# Drop rows with nan values
df = df.dropna()
print(df)

df['Number of ratings'] = df['Number of ratings'].astype(float)
df[column_names_tags]  = df[column_names_tags].astype(float)
df[column_names_tags] = df[column_names_tags].div(df['Number of ratings'], axis=0)
print(df)

# Normalize columns 1 through 8 (assuming zero-based index for columns)
columns_to_normalize = df.columns[:7]
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# Seperate predictor and outcome
pepper = df.iloc[:,3]
predictors = df.drop(df.columns[3], axis=1)
print(pepper)
print(predictors)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    predictors, pepper, test_size=0.2, random_state=rng
)

print(X_train)
print(y_train)

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Efficiently create and display the DataFrame
results = pd.DataFrame({'Predictions': y_pred, 'Probabilities': y_prob})
print(results.tail())  # Display a few rows

print(results[results['Predictions'] == 1].min())
print(results[results['Predictions'] == 0].max())

THRESHOLD = 0.44 
# THRESHOLD = optimal_threshold
y_pred_new = (y_prob > THRESHOLD).astype(int)

class_report = classification_report(y_test, y_pred) #y_pred_new
print(class_report)

print(log_reg.coef_)
print(np.exp(log_reg.coef_)) # These are huge. But careful, we scaled our X variables (between 0 and 1). Still interpretable over fractional units.

print(log_reg.intercept_)
print(np.exp(log_reg.intercept_))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred) #y_pred_new
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["0 (No Pepper)", "1 (Pepper)"], 
            yticklabels=["0 (No Pepper)", "1 (Pepper)"])

# Add title and labels
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.legend()