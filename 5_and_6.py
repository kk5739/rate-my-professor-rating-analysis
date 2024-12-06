import pandas as pd
import sqlite3
import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Set random seed
rng = np.random.default_rng(17010868) 

def main():

    db_path = "rpm.db"
    num_csv_path = "rmpCapstoneNum.csv"
    qual_csv_path = "rmpCapstoneQual.csv"
    tag_csv_path = "rmpCapstoneTags.csv"
    
    column_names = ["Average Rating", 
                "Average Difficulty", 
                "Number of ratings",
                "Received a 'pepper'?",
                "The proportion of students that said they would take the class again",
                "The number of ratings coming from online classes",
                "Male",
                "Female"]
    
    # Only create the database if it doesn't exist
    if not os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df1 = pd.read_csv(num_csv_path, header = None, names = column_names)
        
        df1.iloc[:,0].to_sql(name="rating", con=conn, if_exists="replace", index=False)
        df1.iloc[:,1].to_sql(name="difficulty", con=conn, if_exists="replace", index=False)
        df1.iloc[:,2].to_sql(name="num_ratings", con=conn, if_exists="replace", index=False)
        df1.iloc[:,3].to_sql(name="pepper", con=conn, if_exists="replace", index=False)
        df1.iloc[:,4].to_sql(name="retake", con=conn, if_exists="replace", index=False)
        df1.iloc[:,5].to_sql(name="online", con=conn, if_exists="replace", index=False)
        df1.iloc[:,6].to_sql(name="male", con=conn, if_exists="replace", index=False)
        df1.iloc[:,7].to_sql(name="female", con=conn, if_exists="replace", index=False)
        
        conn.close()
        print("Database created and data inserted.")
    else:
        pass
        
if __name__ == "__main__":
   main()


def get_males_df():
    conn = sqlite3.connect("rpm.db")
    males = pd.read_sql("SELECT * FROM male", conn)
    conn.close()
    return males

def get_females_df():
    conn = sqlite3.connect("rpm.db")
    females = pd.read_sql("SELECT * FROM female", conn)
    conn.close()
    return females

def get_difficulty_df():
    conn = sqlite3.connect("rpm.db")
    difficulty = pd.read_sql("SELECT * FROM difficulty", conn)
    conn.close()
    return difficulty


males = get_males_df()
females = get_females_df()
difficulty = get_difficulty_df()

df = pd.concat([difficulty, males, females], axis=1)

male_diff = df[df['Male'] == 1]['Average Difficulty']
female_diff = df[df['Female'] == 1]['Average Difficulty']

# # Q-Q plot
# fig, ax = plt.subplots(figsize=(6, 6))
# stats.probplot(female_diff, dist="norm", plot=ax)

# # Customize the plot
# ax.get_lines()[1].set_color('red')  # Make the reference line red
# ax.set_title("Q-Q Plot")
# ax.set_xlabel("Theoretical Quantiles")
# ax.set_ylabel("Sample Quantiles")

# Show the plot
plt.show()

test = stats.mannwhitneyu(male_diff, female_diff, alternative="two-sided")
print(test)
###############################################################################
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

male_diff_array = male_diff.to_numpy()
female_diff_array = female_diff.to_numpy()

effect_size = bootstrap(male_diff_array,female_diff_array)
#female_means = bootstrap(female_diff_array)

# Plot the distribution of sample means


# Show the plot
plt.show()

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

# Effect Size (Cohnen's d is not the best. May be Cliff delta is better. But
# because we have a lot of data (power), we are good)
mean1 = np.mean(male_diff_array)
mean2 = np.mean(female_diff_array)
std1 = np.std(male_diff_array)
std2 = np.std(female_diff_array)

numerator = mean1 - mean2
denominator = np.sqrt((std1**2)/2 + (std2**2)/2) # pooled std
d = numerator/denominator
print(d)