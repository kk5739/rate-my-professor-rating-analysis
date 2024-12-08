import pandas as pd
import numpy as np

rng = np.random.default_rng(17010868) 
rng = rng.integers(1,1000000)


def get_numerical(q): 
    column_names_numerical = ["Average Rating", 
                "Average Difficulty", 
                "Number of ratings",
                "Received a 'pepper'?",
                "% students would retake",
                "# Onlne class ratings",
                "Male",
                "Female"]
    df = pd.read_csv('rmpCapstoneNum.csv', header = None, names = column_names_numerical)
    
    if q == 7:
        df = df.dropna()
        df = df[df['Number of ratings'] > np.mean(df['Number of ratings'])]
    else:
        df = df.dropna(subset = ['Number of ratings'])
        df = df[df['Number of ratings'] > np.mean(df['Number of ratings'])]
        
    return df, column_names_numerical

column_names_tags = ["Tough grader", 
                 "Good feedback", 
                 "Respected", 
                 "Lots to read", 
                 "Participation matters", 
                 "Don’t skip class or you will not pass", 
                 "Lots of homework", 
                 "Inspirational", 
                 "Pop quizzes!", 
                 "Accessible", 
                 "So many papers", 
                 "Clear grading", 
                 "Hilarious",
                 "Test heavy", 
                 "Graded by few things", 
                 "Amazing lectures", 
                 "Caring", 
                 "Extra credit", 
                 "Group projects", 
                 "Lecture heavy"]