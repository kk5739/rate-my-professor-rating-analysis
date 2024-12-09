import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from preprocessing import get_numerical, get_tags
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)

# Set random seed
rng = np.random.default_rng(17010868) 
rng = rng.integers(1,1000000)

# Get numerical data and tags data
df_numerical, column_names_numerical = get_numerical(10)
df_tags, column_names_tags = get_tags(10)

# Combine them into one df
df = pd.concat([df_numerical, df_tags], axis=1)
df = df[df['Number of ratings'] > np.mean(df['Number of ratings'])]

# Drop rest of rows with nan values
df = df.dropna()

# Normalize tags
df['Number of ratings'] = df['Number of ratings'].astype(float)
df[column_names_tags]  = df[column_names_tags].astype(float)
df[column_names_tags] = df[column_names_tags].div(df['Number of ratings'], axis=0)

# Normalize columns 1 through 8, so that they have same range as tag data (0-1)
columns_to_normalize = df.columns[:7]
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# Seperate predictor and outcome
pepper = df.iloc[:,3]
predictors = df.drop(df.columns[3], axis=1)

sns.countplot(x="Received a 'pepper'?", data=pd.DataFrame(pepper), palette='Set2')
plt.title('Class Distribution')
plt.xlabel('No Pepper Received (0) or Pepper Received (1)')
plt.ylabel('Count')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    predictors, pepper, test_size=0.2, random_state=rng
)

# Fit logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)
y_prob = log_reg.predict_proba(X_test)[:, 1]

# Generate classification report
class_report = classification_report(y_test, y_pred) #y_pred_new
print(class_report)

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