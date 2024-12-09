import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from preprocessing import get_numerical, get_qual
import pandas as pd
import seaborn as sns
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.oneway import anova_oneway

# Set random seed
rng = np.random.default_rng(17010868) 

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

