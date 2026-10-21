# %%

# Library for Data Manipulation
import numpy as np
import pandas as pd

# %%
# Library for Statistical Modelling
from sklearn.preprocessing import LabelEncoder

# %%
employee_data = pd.read_csv(r"E:\Project 2\HR analytic system\DATASET\IBM-HR-Analytics-Employee dataset.csv")

# %%
# Print top 5 rows in the dataframe.
employee_data.head().style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})

# %%
# Print bottom 5 rows in the dataframe.
employee_data.tail().style.set_properties(**{'background-color': '#E9F6E2','color': 'black','border-color': '#8b8c8c'})

# %%
# Print the shape of the DataFrame
print("The shape of data frame:", employee_data.shape)
# Print the length (number of rows) of the DataFrame
print("Number of Rows in the dataframe:", len(employee_data))
# Print the number of columns in the DataFrame
print("Number of Columns in the dataframe:", len(employee_data.columns))

# %%

print("Column labels in the dataset in column order:")
for column in employee_data.columns:
    print(column)

# %%

# Print the Long summary of the dataframe by setting verbose = True
# Check for Non-Null or Nan Nalues in the dataset.
print(employee_data.info(verbose = True))

# %%


# %% [markdown]
# Inference:
# 
# There are only 26 Numerical Attributes in the dataset.
# 
# On the other hand we have 9 Categorical Attributes.

# %% [markdown]
# 

# %%
employee_data.select_dtypes(np.number).sample(5).style.set_properties(**{'background-color': '#E9F6E2',
                                                              'color': 'black','border-color': '#8b8c8c'})

# %% [markdown]
# 
# Inference:
# 
# Some of the numerical features are storing cateegories labelled in numbers.
# 
# So for better analysis we will replace those labelled numerical values with appropriate categorical values.

# %% [markdown]
#  ###  Labelling Categories in Numerical Feature

# %% [markdown]
# 


