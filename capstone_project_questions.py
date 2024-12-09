#### Lets start by importing all libraries and dependencies
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_numerical , get_tags, get_qual
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression,LogisticRegression,Ridge, Lasso
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report)
from scipy.stats import ks_2samp
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.oneway import anova_oneway

## Importing the random seed attrivuted to N Number for Nikos Prasinos (np3106)
rng = np.random.default_rng(17010868)
seed = 17010868
np.random.seed(17010868)

#%%
####### Question 1 #######
#%%
# Importing and Preprocessing
rmp_df, columns = get_numerical(1)

# Subsetting for males and females
# Checking that males and females have both 1 simultaneously

rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==1)]
# There are rows where they have both females and males meaning that they are "undecided"

# creating df for males and females
male_rating_df = rmp_df[(rmp_df['Male']==1) & (rmp_df['Female']==0)]
female_rating_df = rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==0)]

# Comparing distributions with histograms

plt.figure(figsize=(8,4))
plt.hist(male_rating_df['Average Rating'], alpha=0.2, color='blue')
plt.hist(female_rating_df['Average Rating'], alpha=0.2, color='red')
plt.ylabel('freq')
plt.xlabel('avg_rating')
plt.title('Histogram comparison between Male and Female Avg Rating')
plt.show()

# Not normally dist. So perform a mann whitney
stat, pvalue = stats.mannwhitneyu(male_rating_df['Average Rating'], female_rating_df['Average Rating'], alternative='greater')
print(stat, pvalue)
# Ergo, we reject the null and conclude they are different

# Plotting a violin plot to visualize differences
# Create a new 'Gender' column based on the logic
def determine_gender(row):
    if (row['Male'] == 1) & (row['Female'] == 0):
        return 'Male'
    elif (row['Female'] == 1) & (row['Male'] == 0):
        return 'Female'
    else:
        return 'Not Identified'


rmp_df['Gender'] = rmp_df.apply(determine_gender, axis=1)

plt.figure(figsize=(10, 6))
sns.violinplot(data=rmp_df, x='Gender', y='Average Rating', palette='muted', inner='quartile')
plt.title('Violin Plot of Average Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.show()

#%%
####### Question 2 #######

rng = np.random.default_rng(17010868)

filtered_df_num, num_columns = get_numerical(2)

print("\nH0 (KS Test): Male and female ratings come from the same distribution.")
print("H1 (KS Test): Male and female ratings come from different distributions.")

# Separate normalized data by professor's gender
male_ratings = filtered_df_num[(filtered_df_num["Male"] == 1) & (filtered_df_num["Female"] == 0)]["Average Rating"]
female_ratings = filtered_df_num[(filtered_df_num["Female"] == 1) & (filtered_df_num["Male"] == 0)]["Average Rating"]

# Bootstrap 
def bootstrap(arr1, arr2):
    # Set number of experiments
    num_experiments = int(1e4) # 10000 runs

    # Set number of samples per experiment
    num_samples1 = len(arr1) # Sample 2000 values in each experiment
    num_samples2 = len(arr2) # Sample 2000 values in each experiment


    # Store the sample means in a list
    bootstrapped_var1 = []
    bootstrapped_var2 = []
    # First let's try sampling once
    for i in range(num_experiments):
        # Collect 2000 samples
        indices1 = rng.integers(low=0, high=num_samples1, size=num_samples1)
        indices2 = rng.integers(low=0, high=num_samples2, size=num_samples2)
        
        # print(indices)
        sampled_ratings1 = arr1[indices1]
        sampled_ratings2 = arr2[indices2]
        # print(sampled_ratings)
        # Find the sample mean
        bootstrapped_var1.append(np.var(sampled_ratings1))
        bootstrapped_var2.append(np.var(sampled_ratings2))
    return bootstrapped_var1, bootstrapped_var2

var1, var2 = bootstrap(male_ratings.to_numpy(), female_ratings.to_numpy())

# Plot distribution 
plt.figure(figsize=(8,4))
plt.hist(var1, alpha=0.2, color='blue')
plt.hist(var2, alpha=0.2, color='red')
plt.ylabel('freq')
plt.xlabel('rating')
plt.title('Histogram comparison between Male and Female Rating')
plt.show()


# === KS Test for Distribution Differences ===
ks_stat, ks_p = ks_2samp(var1, var2)
print(f"\nKS Test: Statistic={ks_stat}, p-value={ks_p}")

# Interpret KS test results
if ks_p < 0.005:
    print("Reject H0: Distributions are significantly different.")
else:
    print("Fail to reject H0: No significant difference in distributions.")

#%%
######## Question 3 ###########

rmp_df, columns = get_numerical(3)

# Getting male females df

# creating df for males and females
male_rating_df = rmp_df[(rmp_df['Male']==1) & (rmp_df['Female']==0)]
female_rating_df = rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==0)]

# Defining bootstrap sampling function for cohens d
def bootstrap(arr1, arr2):
    # Set number of experiments
    num_experiments = int(1e4) # 10000 runs

    # Set number of samples per experiment
    num_samples1 = len(arr1) # Sample 2000 values in each experiment
    num_samples2 = len(arr2) # Sample 2000 values in each experiment


    # Store the sample means in a list
    bootstrapped_d = []
    # First let's try sampling once
    for i in range(num_experiments):
        # Collect 2000 samples
        indices1 = rng.integers(low=0, high=num_samples1, size=num_samples1)
        indices2 = rng.integers(low=0, high=num_samples2, size=num_samples2)

        # print(indices)
        sampled_ratings1 = arr1[indices1]
        sampled_ratings2 = arr2[indices2]
        # print(sampled_ratings)
        # Find the sample mean
        mean1 = np.mean(sampled_ratings1)
        mean2 = np.mean(sampled_ratings2)
        std1 = np.std(sampled_ratings1)
        std2 = np.std(sampled_ratings2)


        numerator = mean1 - mean2
        denominator = np.sqrt((std1**2)/2 + (std2**2)/2) # pooled std
        d = numerator/denominator
        bootstrapped_d.append(d)
    return bootstrapped_d

effect_size = bootstrap(male_rating_df['Average Rating'].to_numpy(),female_rating_df['Average Rating'].to_numpy())

# Computing the CI at alpha 0.05 for the effect of bias avg rating means
lower_confidence_bound_male = np.percentile(effect_size, 2.5)
upper_confidence_bound_male = np.percentile(effect_size, 97.5)

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.hist(effect_size, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

# Calculate population mean estimate from bootstrap
effect_size_mean = np.mean(effect_size)

# Add confidence bounds as vertical lines
plt.axvline(lower_confidence_bound_male, color='r', linestyle='dashed', linewidth=1.5, label='2.5% Bound (Male)')
plt.axvline(upper_confidence_bound_male, color='r', linestyle='dashed', linewidth=1.5, label='97.5% Bound (Male)')
plt.axvline(effect_size_mean, color='y', linestyle='solid', linewidth=1.5, label='Mean estimate (Male)')


# Adding labels and title
plt.xlabel('Sample Means from Bootstrapping')
plt.ylabel('Probability')
plt.title('Probability Distribution')

# Add legend
plt.legend()

# Show the plot
plt.show()

# Calculate 95% CI for mean differences
ci_mean_diff = np.percentile(effect_size, [2.5, 97.5])
effect_size_mean_diff = np.mean(effect_size)

# Print Results
print(f"Mean Difference: {effect_size_mean_diff}, 95% CI: {ci_mean_diff}")

# Similarly we do a boostrap but this time for the variances.
# Here we looked at the differences as cohens d focuses on effect sizes for the difference in means
# Here we just look at the differences

def bootstrap_variance_difference(arr1, arr2, num_experiments=10_000):
    boot_var_diff = []
    n1 = len(arr1)
    n2 = len(arr2)
    
    for i in range(num_experiments):
        # Resample indices for each array
        indices1 = rng.integers(low=0, high=n1, size=n1)
        indices2 = rng.integers(low=0, high=n2, size=n2)

        sampled1 = arr1[indices1]
        sampled2 = arr2[indices2]

        # Compute variances
        var1 = np.var(sampled1, ddof=1)  # ddof=1 for sample variance
        var2 = np.var(sampled2, ddof=1)

        # Difference in variance
        diff = var1 - var2
        boot_var_diff.append(diff)
    
    return np.array(boot_var_diff)

# Example usage
boot_var_diff = bootstrap_variance_difference(male_rating_df['Average Rating'].to_numpy(),female_rating_df['Average Rating'].to_numpy())

# Compute 95% CI
lower_bound = np.percentile(boot_var_diff, 2.5)
upper_bound = np.percentile(boot_var_diff, 97.5)

print("95% CI for difference in variances: [{:.4f}, {:.4f}]".format(lower_bound, upper_bound))

plt.figure(figsize=(8, 5))
plt.hist(boot_var_diff, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

mean_diff = np.mean(boot_var_diff)
plt.axvline(mean_diff, color='y', linestyle='solid', linewidth=1.5, label='Mean estimate')

plt.axvline(lower_bound, color='r', linestyle='dashed', linewidth=1.5, label='2.5% Bound')
plt.axvline(upper_bound, color='r', linestyle='dashed', linewidth=1.5, label='97.5% Bound')

plt.xlabel('Bootstrapped Differences in Variance')
plt.ylabel('Density')
plt.title('Distribution of Variance Differences')
plt.legend()
plt.show()

# Calculate 95% CI for mean differences
ci_mean_diff = np.percentile(effect_size, [2.5, 97.5])
effect_size_mean_diff = np.mean(effect_size)

# Print Results
print(f"Mean Difference: {effect_size_mean_diff}, 95% CI: {ci_mean_diff}")

#%%

###### Question 4 ######

filtered_df_num, num_columns = get_numerical(4)
df_tags, tags_columns = get_tags(4)

# Normalize Tag Data 
filtered_df_tags = df_tags.loc[filtered_df_num.index]
filtered_df_tags_normalized = filtered_df_tags.div(filtered_df_num["Number of ratings"], axis=0)
print(filtered_df_tags_normalized.head())

# === Hypotheses ===
print("H0 (Null Hypothesis): There is no gender difference in the tag awarded.")
print("H1 (Alternative Hypothesis): There is a gender difference in the tag awarded.")

# Separate normalized tag data by gender
male_tags = filtered_df_tags_normalized[(filtered_df_num["Male"] == 1) & (filtered_df_num["Female"] == 0)]
female_tags = filtered_df_tags_normalized[(filtered_df_num["Female"] == 1) & (filtered_df_num["Male"] == 0)]

# Perform Mann-Whitney U test for each tag
# Test: MW U test for comparing two independent groups without assuming normality.
results = []
for tag in filtered_df_tags_normalized.columns:
    male_values = male_tags[tag]
    female_values = female_tags[tag]
    
    # Mann-Whitney U Test
    u_stat, p_val = stats.mannwhitneyu(male_values, female_values, alternative='two-sided')
    results.append({"Tag": tag, "U-statistic": u_stat, "p-value": p_val})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Sort by p-value
results_df.sort_values(by="p-value", inplace=True)

# Identify significant tags 
significant_tags = results_df[results_df["p-value"] < 0.005]

# Count significant tags 
num_significant_tags = len(significant_tags)
total_tags = len(filtered_df_tags_normalized.columns)

print(f"\nOut of {total_tags} tags, {num_significant_tags} tag(s) showed a statistically significant gender difference.")

# Identify the most and least gendered tags
most_gendered = results_df.head(3)
least_gendered = results_df.tail(3)

# Display results
print("\nMost Gendered Tags (Lowest p-values):")
print(most_gendered)

print("\nLeast Gendered Tags (Highest p-values):")
print(least_gendered)

# Extract male and female means for each tag
male_means = male_tags.mean()
female_means = female_tags.mean()
tags = male_means.index

# Define bar positions
x = np.arange(len(tags))  # Tag indices for the x-axis
width = 0.35  # Width of the bars

# Create a grouped bar plot
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, male_means, width, label='Male', color='blue', alpha=0.7)
plt.bar(x + width/2, female_means, width, label='Female', color='red', alpha=0.7)

# Add labels, title, and legend
plt.xlabel('Tags', fontsize=12)
plt.ylabel('Mean Normalized Tag Frequency', fontsize=12)
plt.title('Comparison of Tag Frequencies Between Male and Female Professors', fontsize=16)
plt.xticks(x, tags, rotation=45, fontsize=10)
plt.legend()

# Highlight significant tags (optional)
for i, tag in enumerate(tags):
    if tag in significant_tags["Tag"].values:
        plt.text(i, max(male_means[tag], female_means[tag]) + 0.01, '*', 
                 ha='center', va='bottom', fontsize=14, color='black')

plt.tight_layout()
plt.show()


#%%
###### Question 5 #########

# Get numerical data
df,column_names_numerical = get_numerical(5)

# Keep male and female only
male_diff = df[(df['Male'] == 1) & (df['Female'] == 0)]['Average Difficulty']
female_diff = df[(df['Female'] == 1) & (df['Male'] == 0)]['Average Difficulty']

# plot on the same figure the histograms of the two groups
plt.figure(figsize=(8,4))
plt.hist(male_diff, alpha=0.4, bins=30, color='blue', label = 'Male')
plt.hist(female_diff, alpha=0.4, bins=30, color='red', label = 'Female')
plt.legend()
plt.ylabel('Freq')
plt.xlabel('Average Difficulty')
plt.title('Histogram comparison between Male and Female Avg Difficulty')
plt.show()

# Mann Whitney U test (two sided)
test = stats.mannwhitneyu(male_diff, female_diff, alternative="two-sided")
print(test)



#%%
###### Question 6 ######
# Get numerical data
df,column_names_numerical = get_numerical(5)

# Keep male and female only
male_diff = df[(df['Male'] == 1) & (df['Female'] == 0)]['Average Difficulty']
female_diff = df[(df['Female'] == 1) & (df['Male'] == 0)]['Average Difficulty']


def bootstrap(arr1, arr2):
    # Set number of experiments
    num_experiments = int(1e4) # 10000 runs

    # Set number of samples per experiment
    num_samples1 = len(arr1) 
    num_samples2 = len(arr2) 

    bootstrapped_d = []
    for i in range(num_experiments):
        indices1 = rng.integers(low=0, high=num_samples1, size=num_samples1)
        indices2 = rng.integers(low=0, high=num_samples2, size=num_samples2)
        
        # get the subsamples of the two groups
        sampled_ratings1 = arr1[indices1]
        sampled_ratings2 = arr2[indices2]
        
        # Calculate cohen's d based on the samples and add to bootstrap
        mean1 = np.mean(sampled_ratings1)
        mean2 = np.mean(sampled_ratings2)
        std1 = np.std(sampled_ratings1)
        std2 = np.std(sampled_ratings2)
        

        numerator = mean1 - mean2
        denominator = np.sqrt((std1**2)/2 + (std2**2)/2) # pooled std
        d = numerator/denominator
        
        bootstrapped_d.append(d)
    return bootstrapped_d

# Transform pandas df to numpy array
male_diff_array = male_diff.to_numpy()
female_diff_array = female_diff.to_numpy()

# Calculate effect size (Cohen's d)
effect_size = bootstrap(male_diff_array,female_diff_array)


# 95% confidence intervals
lower_confidence_bound = np.percentile(effect_size, 2.5)
upper_confidence_bound = np.percentile(effect_size, 97.5)

# Plot the histogram of bootstrapped effect size
plt.figure(figsize=(8, 5))
plt.hist(effect_size, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

# Calculate population mean estimate from bootstrap
effect_size_mean = np.mean(effect_size)

# Add confidence bounds as vertical lines
plt.axvline(lower_confidence_bound, color='r', linestyle='dashed', linewidth=1.5, label='2.5% Bound')
plt.axvline(upper_confidence_bound, color='r', linestyle='dashed', linewidth=1.5, label='97.5% Bound')
plt.axvline(effect_size_mean, color='y', linestyle='solid', linewidth=1.5, label='Mean effect size estimate')


# Adding labels and title
plt.xlabel(r"Cohen's d")
plt.ylabel('Probability Density')
plt.title('Bootstrapped Effect Size Distribution')

# Add legend
plt.legend()

# Show the plot
plt.show()

print("Effect size: ", np.mean(effect_size))
print("95% Confidence Intervals: [", lower_confidence_bound,",",upper_confidence_bound,"]")

#%%

######## Question 7 #######

# Get numerical data
df,column_names_numerical = get_numerical(7)
# Drop rest of na
df_cleaned = df.dropna()
 
avg_rating = df_cleaned.iloc[:,0]
predictors = df_cleaned.iloc[:,1:]

# Compute the correlation matrix
corr_matrix = predictors.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))  # Optional: Adjust the figure size for better readability
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

# Train-Test Split (80-20)
X_train, X_val, y_train, y_val = train_test_split(predictors, avg_rating, test_size=0.2, random_state=seed)

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

plt.tight_layout()

plt.show()

#%%
####### Question 8 #########


filtered_df_num, num_columns = get_numerical(8)
df_tags, tags_columns = get_tags(8)

# Normalize Tag Data 
filtered_df_tags = df_tags.loc[filtered_df_num.index]
filtered_df_tags_normalized = filtered_df_tags.div(filtered_df_num["Number of ratings"], axis=0)
print(filtered_df_tags_normalized.head())


# === Prepare Data ===
# Predictors: Normalized Tags
X = filtered_df_tags_normalized

# Target: Average Rating
y = filtered_df_num["Average Rating"]

# Train-test split (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) ### N_Number

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

#%%
######## Question 9 ##########
rmp_df, columns = get_numerical(9)

rmp_tags_df , tags_col = get_tags(9)

# Normalize the tags by the number of ratings
rmp_tags_normalized = rmp_tags_df.div(rmp_df["Number of ratings"], axis=0)

avg_diff_tags_df = pd.merge(rmp_df['Average Difficulty'], rmp_tags_normalized, how='left', left_index=True, right_index= True)
avg_diff_tags_df.head()

print(avg_diff_tags_df.isna().sum())
# No nulls

# Addressing multicolinearity concerns
# Plotting correlation to see
plt.figure(figsize=(15,15))
corr_matrix = avg_diff_tags_df.corr()
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm')
plt.title('Corr Matrix')
plt.show()

X = avg_diff_tags_df.drop(columns=['Average Difficulty'])
y = avg_diff_tags_df['Average Difficulty']
# Using correct seed
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=seed)

lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

r2 = r2_score(y_test,y_pred)
print(f"R^2: {r2}")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")

coefs = pd.DataFrame({
    'Feature': avg_diff_tags_df.columns[1:],
    'coefficient':lr.coef_
})

coefs = coefs.sort_values(by='coefficient', ascending= False)

print(coefs)


#%%
########## Question 10 ###########

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
plt.xlabel('No Pepper Received (0) or Pepper Received(1)')
plt.ylabel('Count')
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    predictors, pepper, test_size=0.2, random_state=seed
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
plt.show()

#%%

######## Extra Credit ##########
# Get numerical data
df_numerical,column_names_numerical = get_numerical(11)
df_qual, column_names_qual = get_qual(11)

df = pd.concat([df_numerical, df_qual], axis=1)
df=df.dropna()
df[df['Number of ratings'] > np.mean(df['Number of ratings'])]

major_ratings = df.groupby('Major')['Number of ratings'].sum()

# Sort in descending order to find the majors with the most ratings
major_ratings_sorted = major_ratings.sort_values(ascending=False)

top_majors = major_ratings_sorted.head(3)  # Top 5 majors

# Seperate the top majors into 5 groups and get the average ratings of each major
groups = [df[df['Major'] == major]['Average Difficulty'] for major in top_majors.index.tolist()]

# Create seaborn plot to visualize data
data = pd.DataFrame({
    'avg_difficulty': pd.concat(groups, ignore_index=True),
    'Major': (
        [top_majors.index.tolist()[0]] * len(groups[0]) +
        [top_majors.index.tolist()[1]] * len(groups[1]) +
        [top_majors.index.tolist()[2]] * len(groups[2])
    )
})


palette = sns.color_palette('bright', n_colors=3)

g = sns.histplot(
    data=data,
    x='avg_difficulty',
    hue='Major',
    kde=True,
    element='step',
    alpha=0.3,
    palette=palette  
)


handles, labels = g.get_legend_handles_labels()

if not handles:
    handles = []
    labels = data['Major'].unique().tolist()
    for i, major in enumerate(labels):
        # Use the predefined palette for colors
        handles.append(plt.Line2D([0], [0], color=palette[i], lw=2))

plt.legend(handles=handles, labels=labels, fontsize=10, title_fontsize=12, loc='best')


plt.title('Histogram Comparison Between Majors (Avg Difficulty)', fontsize=14)
plt.xlabel('Average Difficulty', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.show()

#Check for homogeneity of variances (Levene's test)
stat, p = stats.levene(*groups)
print(p)

# Perform Welch's one-way ANOVA
result = anova_oneway([*groups], use_var="unequal")
p_value = result.pvalue

print("p-value:", p_value)

if p_value < 0.005:
    print("There is a significant difference among the groups.")
    # Do a Mann-Whitney U test to check if the 
    # average difficulty is higher in math or biology 
    test1 = stats.mannwhitneyu(groups[0],groups[2], alternative='less')
    print(test1.pvalue)
else:
    print("No significant difference found among the groups.")