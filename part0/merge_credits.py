import pandas as pd

df1 = pd.read_csv('credits_1.csv')
df2 = pd.read_csv('credits_2.csv')

merge_dfs = pd.concat([df1, df2], axis=0)

merge_dfs.to_csv('credits.csv', index=False)
