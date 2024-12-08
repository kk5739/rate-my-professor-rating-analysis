import pandas as pd
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

# === Normalize Tag Data ===
# Create a normalized version of the tags dataframe
df_tags_normalized = df_tags.div(df_num["Number of Ratings"], axis=0)

# Separate normalized data by gender
male_tags = df_tags_normalized[df_num["Male"] == 1]
female_tags = df_tags_normalized[df_num["Female"] == 1]


## Q2. Is there a gender difference in the spread (variance/dispersion) 
## of the ratings distribution? Again, it is advisable to consider 
## the statistical significance of any observed gender differences 
## in this spread.

print("H0 (Null Hypothesis): There is a gender difference in the spread of the ratings distribution.")
print("H1: There is a no gender difference in the spread of the ratings distribution.")

# Set a threshold for the minimum number of ratings
k = 5.37  # This df_num["Number of Ratings"].mean()

# Step 1: Apply the threshold
filtered_data = df_num[df_num["Number of Ratings"] >= k]

# Step 2: Drop rows with missing average ratings
filtered_data = filtered_data.dropna(subset=["Average Rating"])

# Step 3: Separate ratings by gender
male_ratings = filtered_data[filtered_data["Male"] == 1]["Average Rating"]
female_ratings = filtered_data[filtered_data["Female"] == 1]["Average Rating"]

# Step 4: Calculate variances for each gender
male_variance = np.var(male_ratings, ddof=1)  # Sample variance
female_variance = np.var(female_ratings, ddof=1)

# Output results
print(f"Male Variance: {male_variance}")
print(f"Female Variance: {female_variance}")

## KS Test for Distribution (sensitive to differences in shape, spread, and median).

from scipy.stats import ks_2samp

# Perform the KS test
ks_stat, p_value = ks_2samp(male_ratings, female_ratings)

# Output results
print(f"KS Statistic: {ks_stat}")
print(f"p-value: {p_value}")

# Interpret the results
if p_value < 0.0005:
    print("We are dropping the assumed null hypothesis, H1 is true. There is a significant difference in the spread of ratings between male and female professors.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the spread of ratings between male and female professors.")

########################
### Chi-sqaure Test for Variance comparison

#from scipy.stats import chi2

# Variance and sample sizes
#male_variance = np.var(male_ratings, ddof=1)  
#female_variance = np.var(female_ratings, ddof=1)
#male_n = len(male_ratings)
#female_n = len(female_ratings)

# Chi-Square statistic
#chi2_stat = (male_n - 1) * male_variance / female_variance

# Degrees of freedom
#df_male = male_n - 1
#df_female = female_n - 1

# p-value from chi-square distribution
#p_value = 1 - chi2.cdf(chi2_stat, df=df_male)

#print(f"Chi-Square Statistic: {chi2_stat}")
#print(f"p-value: {p_value}")

# Interpret the results
#if p_value < 0.005:
#    print("We are dropping the assumed null hypothesis, H1 is true. There is a significant difference in the spread of ratings between male and female professors.")
#else:
#    print("Fail to reject the null hypothesis. There is no significant difference in the spread of ratings between male and female professors.")
#####################

### MW U test distribution (median-based comparison).
from scipy.stats import mannwhitneyu

# Perform Mann-Whitney U Test
u_stat, p_value = mannwhitneyu(male_ratings, female_ratings, alternative="two-sided")

# Output results
print(f"Mann-Whitney U Statistic: {u_stat}")
print(f"p-value: {p_value}")

# Interpret the results
if p_value < 0.0005:
    print("We are dropping the assumed null hypothesis, H1 is true. There is a significant difference in the spread of ratings between male and female professors.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference in the spread of ratings between male and female professors.")


