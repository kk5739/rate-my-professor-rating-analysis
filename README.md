# Rate My Professor: Review Analysis & Insights

## Overview

This project explores and analyzes data from Rate My Professors (RMP) to uncover patterns in student feedback, instructor ratings, and tag-based sentiment. By combining qualitative and quantitative data, the project aims to generate meaningful insights into teaching effectiveness, perceived difficulty, gender bias, and predictors of overall ratings.

## Objective

- Assess gender bias in rating distributions, variability, and tags
- Determine key factors influencing average ratings and difficulty
- Use regression and classification models to predict outcomes like ratings and "pepper" status
- Normalize and merge multiple data sources into an SQL database for queryability

## Data

This project uses three primary datasets:

- `rmpCapstoneNum.csv`: Numerical ratings (e.g., overall, clarity, difficulty)
- `rmpCapstoneQual.csv`: Free-text student reviews
- `rmpCapstoneTags.csv`: Tags like "Tough Grader", "Hilarious"

An SQLite database `rpm.db` consolidates the datasets for advanced querying.

## Key Analyses & Models

1. **Gender Bias Testing**

   - Mann-Whitney U and KS tests reveal statistically significant differences in ratings and variance between male and female professors.
   - Bootstrapped effect sizes confirm small but notable gender bias.

2. **Tag-Based Gender Analysis**

   - Chi-squared tests indicate significant gender differences in 11 of 20 tags.
   - Most gendered tags: "Hilarious", "Amazing Lectures", "Caring".

3. **Regression Modeling**

   - Predicting `Average Rating`:

     - **From numerical data** → R² = 0.8410, RMSE = 0.3306
     - **From tags** → R² = 0.7339, RMSE = 0.4842
     - Most predictive factor: % of students who would retake the course

   - Predicting `Average Difficulty`:

     - From tags → R² = 0.579, RMSE = 0.518
     - Most predictive tag: "Tough Grader" (+1.736 coefficient)

4. **Classification Modeling**

   - Predicting "Pepper" status using numerical + tag data
   - Logistic Regression
   - Accuracy: 76%, AUROC: 0.83

5. **Extra Credit**: Statistical analysis of perceived difficulty across top 3 majors using Levene’s test, Welch’s ANOVA, and Mann-Whitney U test.

## Project Structure

```
Rate My Professor Review Analysis
├── data/
│   ├── rmpCapstoneNum.csv
│   ├── rmpCapstoneQual.csv
│   └── rmpCapstoneTags.csv
├── __pycache__/
│   └── preprocessing.cpython-3xx.pyc
├── rmp_final_code.py                  # Full preprocessing and modeling script
├── rmp_final_report.pdf               # Formal report detailing analyses
├── IDS capstone project spec sheet.pdf
├── rpm.db                             # Consolidated SQLite database
```

## Technologies Used

- Python 3.11+
- Pandas, NumPy
- SciPy, Scikit-learn
- SQLite3
- Seaborn, Matplotlib
- Statsmodels

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/kk5739/rate-my-professor-rating-analysis.git
```

2. Set up your Python environment:

```bash
pip install -r requirements.txt  # if available
```

3. Run the main script:

```bash
python rmp_final_code.py
```

## Results Summary

- Male professors receive statistically higher ratings than females (p < 0.001).
- Ratings for female professors show significantly more variability.
- "Tough Grader" tag is consistently linked to lower ratings and higher difficulty.
- Retake likelihood (% students would retake) is the most predictive variable overall.
- Classification model for "Pepper" status yields high AUROC of 0.83.

## License

This project is part of NYU's IDS Capstone (Fall 2024) and is intended for academic use only.

## Authors

Capstone Group #52 – NYU Intro to Data Science

- Kyeongmo Kang 
- Nikolas Prasinos
- Alexander Pegot-Ogier 

For questions or reuse inquiries, please open an issue or contact a contributor.

