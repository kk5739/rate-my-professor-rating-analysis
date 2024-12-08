## Question 1
#%%
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_numerical
import seaborn as sns
from scipy import stats
#%%
# Importing and Preprocessing
rmp_df, columns = get_numerical(1)
#%%
# Subsetting for males and females
# Checking that males and females have both 1 simultaneously

rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==1)]
# There are rows where they have both females and males meaning that they are "undecided"
#%%
# creating df for males and females
male_rating_df = rmp_df[(rmp_df['Male']==1) & (rmp_df['Female']==0)]
female_rating_df = rmp_df[(rmp_df['Female']==1) & (rmp_df['Male']==0)]

#%%
# Comparing distributions with histograms

plt.figure(figsize=(8,4))
plt.hist(male_rating_df['Average Rating'], alpha=0.2, color='blue')
plt.hist(female_rating_df['Average Rating'], alpha=0.2, color='red')
plt.ylabel('freq')
plt.xlabel('avg_rating')
plt.title('Histogram comparison between Male and Female Avg Rating')
plt.show()
#%%
# Not normally dist. So perform a mann whitney
stat, pvalue = stats.mannwhitneyu(male_rating_df['Average Rating'], female_rating_df['Average Rating'], alternative='greater')
print(stat, pvalue)
# Ergo, we reject the null and conclude they are different
#%%

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
#%%
plt.figure(figsize=(10, 6))
sns.violinplot(data=rmp_df, x='Gender', y='Average Rating', palette='muted', inner='quartile')
plt.title('Violin Plot of Average Ratings by Gender')
plt.xlabel('Gender')
plt.ylabel('Average Rating')
plt.show()




