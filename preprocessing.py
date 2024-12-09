import pandas as pd
import numpy as np

# Get numerical data
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
    
    # drop accordingly
    if q == 7: # drop all nan and keep professors with more ratings than the mean
        df = df.dropna()
        df = df[df['Number of ratings'] > np.mean(df['Number of ratings'])]
    elif q==10: # don't drop anything
        pass 
    else:
        df = df.dropna(subset = ['Number of ratings'])
        df = df[df['Number of ratings'] > np.mean(df['Number of ratings'])]
        
    return df, column_names_numerical

# Get tags
def get_tags(q): 
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
    df = pd.read_csv('rmpCapstoneTags.csv', header = None, names = column_names_tags)

        
    return df, column_names_tags

# Get qualitative data
def get_qual(q): 
    column_names_qual= [
        "Major",
        "University",
        "StateCode"
    ]
    df = pd.read_csv('rmpCapstoneQual.csv', header = None, names = column_names_qual)
    df.dropna()
        
    return df, column_names_qual