# Q3
#%%
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_numerical
import seaborn as sns
from scipy import stats
import numpy as np
#%%
# Importing and Preprocessing
rmp_df, columns = get_numerical(1)

# Getting male females df

# creating df for males and females
male_rating_df = rmp_df[(rmp_df['Male']==1) & (rmp_df['Female']==0)]
female_rating_df = rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==0)]

#%%
# Setting random seed 
rng = np.random.default_rng(17010868)
#%%

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

#%%
effect_size = bootstrap(male_rating_df['Average Rating'].to_numpy(),female_rating_df['Average Rating'].to_numpy())
#%%
# Computing the CI at alpha 0.05 for the effect of bias avg rating means

# Let's calculate the 95% confidence intervals
lower_confidence_bound_male = np.percentile(effect_size, 2.5)
upper_confidence_bound_male = np.percentile(effect_size, 97.5)

# Let's plot these two points on the plot!

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
#%%

# Calculate 95% CI for mean differences
ci_mean_diff = np.percentile(effect_size, [2.5, 97.5])
effect_size_mean_diff = np.mean(effect_size)

# Print Results
print(f"Mean Difference: {effect_size_mean_diff}, 95% CI: {ci_mean_diff}")


#%% 

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


#%%
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
#%%

# Calculate 95% CI for mean differences
ci_mean_diff = np.percentile(effect_size, [2.5, 97.5])
effect_size_mean_diff = np.mean(effect_size)

# Print Results
print(f"Mean Difference: {effect_size_mean_diff}, 95% CI: {ci_mean_diff}")

#%%

