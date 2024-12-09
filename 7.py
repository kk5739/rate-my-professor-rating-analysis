from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import get_numerical

# Set random seed
rng = np.random.default_rng(17010868) 
rng = rng.integers(1,1000000)

# Get numerical data
df,column_names_numerical = get_numerical(7)
# Drop rest of na
df_cleaned = df.dropna()
 
avg_rating = df_cleaned.iloc[:,0]
predictors = df_cleaned.iloc[:,1:]

# Train-Test Split (80-20)
X_train, X_val, y_train, y_val = train_test_split(predictors, avg_rating, test_size=0.2, random_state=rng)

# Standardize the features (scale to zero mean and unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict on training and test data for Linear Regression
y_train_pred_linear = linear_model.predict(X_train_scaled)
y_val_pred_linear = linear_model.predict(X_val_scaled)

# Calculate MSE for Linear Regression
train_mse_linear = mean_squared_error(y_train, y_train_pred_linear)
test_mse_linear = mean_squared_error(y_val, y_val_pred_linear)

print(f"Linear Regression - RMSE: {np.sqrt(test_mse_linear)}")
print(f"Linear Regression - R^2: {r2_score(y_val, y_val_pred_linear)}\n")

# Train a Ridge Regression model with different alpha values
ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_scaled, y_train)
    
# Predict on training and test data for Ridge Regression
y_train_pred_ridge = ridge_model.predict(X_train_scaled)
y_val_pred_ridge = ridge_model.predict(X_val_scaled)

ridge_train_mse= mean_squared_error(y_train, y_train_pred_ridge)
ridge_val_mse = mean_squared_error(y_val, y_val_pred_ridge)

# Check ridge_val_mse for lowest val mse, and pick the corresponding alpha
print(f"Ridge Regression - RMSE: {np.sqrt(ridge_val_mse)}")
print(f"Ridge Regression - R^2: {r2_score(y_val, y_val_pred_ridge)}\n")

# Train a Lasso Regression model with different alpha values
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)

# Predict on training and test data for Lasso Regression
y_train_pred_lasso = lasso_model.predict(X_train_scaled)
y_val_pred_lasso = lasso_model.predict(X_val_scaled)

# Calculate MSE for Lasso Regression
lasso_train_mse = mean_squared_error(y_train, y_train_pred_lasso)
lasso_val_mse = mean_squared_error(y_val, y_val_pred_lasso)

# Check lasso_val_mse for lowest val mse, and pick the corresponding alpha

print(f"Lasso Regression - RMSE: {np.sqrt(lasso_val_mse)}")
print(f"Lasso Regression - R^2: {r2_score(y_val, y_val_pred_lasso)}\n")


# Ridge model coefficients (for the best alpha, e.g., alpha=1.0)
ridge_model = Ridge(alpha=0.01)
ridge_model.fit(X_train_scaled, y_train)
betas_ridge = ridge_model.coef_

# Lasso model coefficients (for the best alpha, e.g., alpha=1.0)
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)
betas_lasso = lasso_model.coef_

# Plot the coefficients (betas) for all models
plt.figure(figsize=(18, 6))

# Ridge Regression Coefficients (for best alpha)
plt.subplot(1, 2, 1)
plt.bar(range(len(betas_ridge)), betas_ridge)
plt.title('Coefficients Ridge (alpha=0.01)')
plt.xticks(
    ticks=range(len(column_names_numerical[1:])),
    labels=column_names_numerical[1:], 
    rotation=45
)
plt.ylabel('Coefficient Value')

# Lasso Regression Coefficients (for best alpha)
plt.subplot(1, 2, 2)
plt.bar(range(len(betas_lasso)), betas_lasso)
plt.title('Coefficients Lasso (alpha=0.01)')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.tight_layout()

plt.show()