import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from preprocessing import get_numerical

# Set random seed
rng = np.random.default_rng(17010868) 

# Get numerical data
df,column_names_numerical = get_numerical(5)

# Keep male and female only
male_diff = df[(df['Male'] == 1) & (df['Female'] == 0)]['Average Difficulty']
female_diff = df[(df['Female'] == 1) & (df['Male'] == 0)]['Average Difficulty']

# plot on the same figure the histograms of the two groups
plt.figure(figsize=(8,4))
plt.hist(male_diff, alpha=0.2, color='blue')
plt.hist(female_diff, alpha=0.2, color='red')
plt.ylabel('freq')
plt.xlabel('avg_difficulty')
plt.title('Histogram comparison between Male and Female Avg Difficulty')
plt.show()

# Mann Whitney U test (two sided)
test = stats.mannwhitneyu(male_diff, female_diff, alternative="two-sided")
print(test)
###############################################################################
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
lower_confidence_bound_male = np.percentile(effect_size, 2.5)
upper_confidence_bound_male = np.percentile(effect_size, 97.5)

# Plot the histogram of bootstrapped effect size
plt.figure(figsize=(8, 5))
plt.hist(effect_size, bins=50, density=True, alpha=0.6, color='b', edgecolor='black')

# Calculate population mean estimate from bootstrap
effect_size_mean = np.mean(effect_size)

# Add confidence bounds as vertical lines
plt.axvline(lower_confidence_bound_male, color='r', linestyle='dashed', linewidth=1.5, label='2.5% Bound')
plt.axvline(upper_confidence_bound_male, color='r', linestyle='dashed', linewidth=1.5, label='97.5% Bound')
plt.axvline(effect_size_mean, color='y', linestyle='solid', linewidth=1.5, label='Mean effect size estimate')


# Adding labels and title
plt.xlabel(r"Cohen's d")
plt.ylabel('Probability Density')
plt.title('Bootstrapped Effect Size Distribution')

# Add legend
plt.legend()

# Show the plot
plt.show()