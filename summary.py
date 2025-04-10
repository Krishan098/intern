import pandas as pd
df=pd.read_csv('dataset.csv')
print(df.describe())
print("columns:",list(df.columns))
print("Total comments: ",len(df))
print("Missing values:",df.isnull().sum())