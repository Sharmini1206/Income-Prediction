import pip
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

df = pd.read_csv(".\\Assets\\adult.csv")
# print (df)

df.education.value_counts()
#print(df.education.value_counts())
df.workclass.value_counts()
#print(df.workclass.value_counts())
df.gender.value_counts()
#print(df.gender.value_counts())
df.occupation.value_counts()
#print(df.occupation.value_counts())

df = pd.concat([df.drop("occupation", axis=1), pd.get_dummies(df.occupation).add_prefix("occupation_")],axis=1)
df = pd.concat([df.drop("workclass", axis=1), pd.get_dummies(df.workclass).add_prefix("workclass_")],axis=1)
df = pd.concat((df.drop("marital-status", axis=1), pd.get_dummies(df['marital-status']).add_prefix("marital-status_")), axis=1)
df = pd.concat([df.drop("relationship", axis=1), pd.get_dummies(df.relationship).add_prefix("relationship_")],axis=1)
df = pd.concat([df.drop("race", axis=1), pd.get_dummies(df.race).add_prefix("race_")],axis=1)
df = pd.concat([df.drop("native-country", axis=1), pd.get_dummies(df["native-country"]).add_prefix("native-country_")],axis=1)
df = df.drop("education", axis=1)
print(df)

df["gender"] = df["gender"].apply(lambda x: 1 if x == "Male" else 0)
df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)
print(df)

df.columns.values
print(df.columns.values)

plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")

plt.show()

correlations = df.corr()["income"].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8*len(df.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
df_dropped = df.drop(cols_to_drop, axis =1)

print(df_dropped)