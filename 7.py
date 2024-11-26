from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1) 
rng = rng.integers(1,1000000)
df = pd.read_csv('rmpCapstoneNum.csv', header = None)

# Row - wise removal
df_cleaned = df.dropna()

avg_rating = df_cleaned.iloc[:,0]
predictors = df_cleaned.iloc[:,1:]

# Step 2: Train-Test Split (80-20)
X_train, X_val, y_train, y_val = train_test_split(predictors, avg_rating, test_size=0.2, random_state=rng)

# Step 3: Standardize the features (scale to zero mean and unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Step 4: Train a Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)

# Predict on training and test data for Linear Regression
y_train_pred_linear = linear_model.predict(X_train_scaled)
y_val_pred_linear = linear_model.predict(X_val_scaled)

# Calculate MSE for Linear Regression
train_mse_linear = mean_squared_error(y_train, y_train_pred_linear)
test_mse_linear = mean_squared_error(y_val, y_val_pred_linear)

print(f"Linear Regression - RMSE: {np.sqrt(test_mse_linear)}")
print(f"Linear Regression - R^2: {r2_score(y_val, y_val_pred_linear)}")

# Step 5: Train a Ridge Regression model with different alpha values
alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]  # Experiment with different alpha values
alphas = [0.01]
ridge_train_mse = []
ridge_val_mse = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)
    
    # Predict on training and test data for Ridge Regression
    y_train_pred_ridge = ridge_model.predict(X_train_scaled)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)
    
    
    ridge_train_mse.append(mean_squared_error(y_train, y_train_pred_ridge))
    ridge_val_mse.append(mean_squared_error(y_val, y_val_pred_ridge))

best_alpha_ridge = 0.1
# Check ridge_val_mse for lowest val mse, and pick the corresponding alpha

print(f"Ridge Regression - RMSE: {np.sqrt(ridge_val_mse)}")
print(f"Ridge Regression - R^2: {r2_score(y_val, y_val_pred_ridge)}")


plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_train_mse, label='Training MSE', marker='o')
plt.plot(alphas, ridge_val_mse, label='Validation MSE', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Ridge Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()


# Step 6: Train a Lasso Regression model with different alpha values
lasso_train_mse = []
lasso_val_mse = []

alphas = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]
alphas = [0.01]

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)
    
    # Predict on training and test data for Lasso Regression
    y_train_pred_lasso = lasso_model.predict(X_train_scaled)
    y_val_pred_lasso = lasso_model.predict(X_val_scaled)
    
    # Calculate MSE for Lasso Regression
    lasso_train_mse.append(mean_squared_error(y_train, y_train_pred_lasso))
    lasso_val_mse.append(mean_squared_error(y_val, y_val_pred_lasso))

# Check lasso_val_mse for lowest val mse, and pick the corresponding alpha

print(f"Lasso Regression - RMSE: {np.sqrt(lasso_val_mse)}")
print(f"Lasso Regression - R^2: {r2_score(y_val, y_val_pred_lasso)}")



best_alpha_lasso = 0.1

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_train_mse, label='Training MSE', marker='o')
plt.plot(alphas, lasso_val_mse, label='Validation MSE', marker='o')
plt.xscale('log')  # Log scale for alpha
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Lasso Regularization on MSE')
plt.legend()
plt.grid(True)
plt.show()


# Step 7: Visualize the betas (coefficients) from all models

# Ridge model coefficients (for the best alpha, e.g., alpha=1.0)
ridge_model = Ridge(alpha=best_alpha_ridge)
ridge_model.fit(X_train_scaled, y_train)
betas_ridge = ridge_model.coef_

# Lasso model coefficients (for the best alpha, e.g., alpha=1.0)
lasso_model = Lasso(alpha=best_alpha_lasso)
lasso_model.fit(X_train_scaled, y_train)
betas_lasso = lasso_model.coef_

# Plot the coefficients (betas) for all models
plt.figure(figsize=(18, 6))


# Ridge Regression Coefficients (for best alpha)
plt.subplot(1, 2, 1)
plt.bar(range(len(betas_ridge)), betas_ridge)
plt.title(f'Coefficients Ridge (alpha={best_alpha_ridge})')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

# Lasso Regression Coefficients (for best alpha)
plt.subplot(1, 2, 2)
plt.bar(range(len(betas_lasso)), betas_lasso)
plt.title(f'Coefficients Lasso (alpha={best_alpha_lasso})')
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')

plt.tight_layout()
plt.show()