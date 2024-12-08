import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


###################################################### Q4. ##################################################################################################

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
