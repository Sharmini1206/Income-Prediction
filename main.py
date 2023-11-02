import pip
import pandas as pd
import numpy
import matplotlib
import sklearn
import seaborn

df = pd.read_csv(".\\Assets\\adult.csv")
# print (df)

df.education.value_counts()
print(df.education.value_counts())
df.workclass.value_counts()
print(df.workclass.value_counts())
df.gender.value_counts()
print(df.gender.value_counts())
df.occupation.value_counts()
print(df.occupation.value_counts())

pd.get_dummies(df.occupation)
print(pd.get_dummies(df.occupation))

