# Q9

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import get_numerical
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

#%%
# importing data
rmp_tag_df = pd.read_csv('data/rmpCapstoneTags.csv',
                         names=['tough_grader','good_feedback','respected','lots_to_read',
                                'participation_matters','dscoywp','lots_homework','inspirationl',
                                "pop_quizes",'accessible','many_papers','clear_grading','hilarious',
                                'test_heavy','graded_by_few_things','amazing_lectures','extra_credit','caring',
                                'group_proj','lecture_heavy'])

rmp_df, columns = get_numerical(1)

#%%
# Normalize the tags by the number of ratings
rmp_tags_normalized = rmp_tag_df.div(rmp_df["Number of ratings"], axis=0)

#%%
avg_diff_tags_df = pd.merge(rmp_df['Average Difficulty'], rmp_tags_normalized, how='left', left_index=True, right_index= True)
avg_diff_tags_df.head()
#%%
print(avg_diff_tags_df.isna().sum())
# No nulls

#%%
# Addressing multicolinearity concerns
# Plotting correlation to see
plt.figure(figsize=(15,15))
corr_matrix = avg_diff_tags_df.corr()
sns.heatmap(corr_matrix,annot=True, cmap='coolwarm')
plt.title('Corr Matrix')
plt.show()

#%%
X = avg_diff_tags_df.drop(columns=['Average Difficulty'])
y = avg_diff_tags_df['Average Difficulty']
# Using correct seed
X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=17010868)

#%%
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
#%%
r2 = r2_score(y_test,y_pred)
print(f"R^2: {r2}")
#%%
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")
#%%
coefs = pd.DataFrame({
    'Feature': avg_diff_tags_df.columns[1:],
    'coefficient':lr.coef_
})

coefs = coefs.sort_values(by='coefficient', ascending= False)

print(coefs)
#%%


