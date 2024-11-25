#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:46:54 2024

@author: kangkyeongmo
"""

import sqlite3
import pandas as pd
import re
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import numpy as np


# General info, Define  column names
column_names = ["Average Rating", 
                "Average Difficulty", 
                "Number of ratings",
                "Received a 'pepper'?",
                "The proportion of students that said they would take the class again",
                "The number of ratings coming from online classes",
                "Male",
                "Female"]
df_num = pd.read_csv("rmpCapstoneNum.csv", header=None, names=column_names)
print(df_num.columns)

# Major|University|States, define column names
column_names2 = ["Major/Field", "University", "US State (2 letter abbreviation)" ]
df_qual = pd.read_csv("rmpCapstoneQual.csv", header=None, names=column_names2)
print(df_qual.columns)

# 3 Tags, define column names
column_names3 = ["Tough grader", 
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
df_tags = pd.read_csv("rmpCapstoneTags.csv", header=None, names=column_names3)
print(df_tags.columns)

## Q8. Build a regression model predicting average ratings from all 
## tags (the ones in the rmpCapstoneTags.csv) file. Make sure to 
## include the R2 and RMSE of this model. Which of these tags is most
## strongly predictive of average rating? Hint: Make sure to address 
## collinearity concerns. Also comment on how this model compares 
## to the previous one.