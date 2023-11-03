import pip
import pandas as pd
import numpy
import matplotlib
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


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

plt.figure(figsize=(18, 12))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")

#plt.show()

correlations = df.corr()["income"].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8*len(df.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
df_dropped = df.drop(cols_to_drop, axis =1)

plt.figure(figsize=(15, 10))
sns.heatmap(df_dropped.corr(), annot=True, cmap="coolwarm")

#plt.show()


# here fnlwgt= Final weight
# The weights on the Current Population Survey (CPS) files are controlled to independent
# estimates of the civilian non-institutional population of the US.
# (fnlwgt featured that How many people belong to that grp)
# that's not much importance to this data. so we can drop that

df = df.drop("fnlwgt", axis = 1)

train_df, test_df = train_test_split(df, test_size=0.2)
print(test_df)
print(train_df)

train_x = train_df.drop("income", axis=1)
train_y = train_df["income"]

test_x = test_df.drop("income", axis=1)
test_y = test_df["income"]

forest = RandomForestClassifier()

e1 = forest.fit(train_x, train_y)
e2 = forest.score(test_x, test_y)

print(e1)
print(e2)
#so 85% of the time accurate we can predict this person is earning more than or less than 50k

forest.feature_importances_
forest.feature_names_in_

importance = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importance = {k: v for k, v in sorted(importance.items(), key=lambda x:x[1], reverse=True)}

print(importance)

#param_grid = {
#    "n_estimators": [50, 100, 250],
#    "max_depth": [5, 10, 30, None],
#   "min_samples_split":[2, 4],
#    "max_features": ["sqrt", "log2"]
#}

param_grid = {
    "n_estimators": [250],
    "max_depth": [5, None],
    "min_samples_split":[2, 4],
    "max_features": ["sqrt", "log2"]
}

grid_search = GridSearchCV(estimator= RandomForestClassifier(),
                           param_grid= param_grid, verbose=10)
grid_search.fit(train_x, train_y)
grid_search.best_estimator_
print(grid_search.best_estimator_)

forest = grid_search.best_estimator_
forest.score(test_x, test_y)

importance = dict(zip(forest.feature_names_in_, forest.feature_importances_))
importance = {k: v for k, v in sorted(importance.items(), key=lambda x:x[1], reverse=True)}
print(importance)