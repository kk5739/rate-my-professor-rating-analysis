import pandas as pd
from scipy.stats import ttest_ind
import os 

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

# === Hypothesis Testing ===
# Perform t-tests for gender differences in each tag
results = []
for tag in tags_columns:
    t_stat, p_val = ttest_ind(male_tags[tag], female_tags[tag], nan_policy='omit')
    results.append({"Tag": tag, "t-statistic": t_stat, "p-value": p_val})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Sort results by p-value
results_df.sort_values(by="p-value", inplace=True)

# Identify statistically significant tags (p < 0.0005)
significant_tags = results_df[results_df["p-value"] < 0.0005]

# === Output Results ===
# Print results for significant tags, most gendered, and least gendered tags
print("Statistically Significant Tags (p < 0.0005):")
print(significant_tags)

print("\nMost Gendered Tags (Lowest p-values):")
print(results_df.head(3))

print("\nLeast Gendered Tags (Highest p-values):")
print(results_df.tail(3))

# === Top Tags by Gender ===
# Calculate the mean normalized tag frequency for each gender
male_tag_means = male_tags.mean()
female_tag_means = female_tags.mean()

# Identify top 3 tags for each gender
print("\nTop 3 Tags for Male Professors:")
print(male_tag_means.sort_values(ascending=False).head(3))

print("\nTop 3 Tags for Female Professors:")
print(female_tag_means.sort_values(ascending=False).head(3))

# === df_qual Availability ===
# Preview qualitative data
print("\nQualitative Data (df_qual) Head:")
print(df_qual.head())


# Hypotheses
print("H0 (Null Hypothesis): There is no gender difference in the tags awarded by students.")
print("H1 (Alternative Hypothesis): There is a gender difference in the tags awarded by students.")

# Normalize tag data by the number of ratings
df_tags_normalized = df_tags.div(df_num["Number of Ratings"], axis=0)

# Filter by gender
male_tags = df_tags_normalized[df_num["Male"] == 1]
female_tags = df_tags_normalized[df_num["Female"] == 1]

# Perform t-tests for gender differences in each tag (comparing the means of two independent groups)
p_values = {}
for tag in df_tags.columns:
    t_stat, p_val = ttest_ind(male_tags[tag], female_tags[tag], nan_policy='omit')
    p_values[tag] = p_val

# Convert p-values to DataFrame
p_values_df = pd.DataFrame(list(p_values.items()), columns=["Tag", "p-value"])
p_values_df.sort_values(by="p-value", inplace=True)

# Identify the most and least gendered tags
most_gendered = p_values_df.head(3)
least_gendered = p_values_df.tail(3)

# Determine number of significant and non-significant tags
significance_threshold = 0.0005
significant_tags = p_values_df[p_values_df["p-value"] < significance_threshold]
non_significant_tags = p_values_df[p_values_df["p-value"] >= significance_threshold]

# Print most and least gendered tags
print("\nMost Gendered Tags (Lowest p-values):")
print(most_gendered)

print("\nLeast Gendered Tags (Highest p-values):")
print(least_gendered)

# Summary and conclusion
print("\n=== Conclusion ===")
total_tags = len(df_tags.columns)
num_significant = len(significant_tags)
num_non_significant = total_tags - num_significant

print(f"Out of {total_tags} tags:")
print(f"- {num_significant} tag(s) showed a statistically significant gender difference (p < {significance_threshold}).")
print(f"- {num_non_significant} tag(s) did not show a statistically significant gender difference.")
