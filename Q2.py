import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

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

###################################################### Q2. ##################################################################################################

# === Hypotheses ===
# Test: KS Test for Comparing the overal distributions  
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
