import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# import tain and test csv
train = pd.read_csv("C:/Users/PruthuviA/Desktop/titanic/csv/train.csv")
test = pd.read_csv("C:/Users/PruthuviA/Desktop/titanic/csv/test.csv")

# combining train and test dataset
# from combining logic can be use to both data set
train_test_data = [train, test]

# get the title from the name
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(
        ' ([A-Za-z]+)\.', expand=False)

# after analysing devide the titles to 4 types as Proportional
title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3, "Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona": 3, "Mme": 3, "Capt": 3, "Sir": 3}

# add column title
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

# drop column name. its not usable
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

# set sex male as 0, female as 1
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

# set age as median to both data sets median is 30
train["Age"].fillna(train.groupby("Title")[
                    "Age"].transform("median"), inplace=True)
test["Age"].fillna(test.groupby("Title")[
                   "Age"].transform("median"), inplace=True)

# after graph analysing devide that age to 4 categories
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[dataset['Age'] > 62, 'Age'] = 4

# after analysing, more people embarked from s, so another 2 null values filled with s
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# define embarked ports for 3 categories
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

# fill missing Fare with median fare for each Pclass
train["Fare"].fillna(train.groupby("Pclass")[
                     "Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")[
                    "Fare"].transform("median"), inplace=True)

# after graph analysing devide that age to 4 categories
for dataset in train_test_data:
    dataset.loc[dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[dataset['Fare'] > 100, 'Fare'] = 3

# after analysing devide cabin to 8 types
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8,
                 "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

# import that cabin numeric types
train["Cabin"].fillna(train.groupby("Pclass")[
                      "Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")[
                     "Cabin"].transform("median"), inplace=True)

# after analysing devide fam size to 11 types as Proportional
train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1
family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6,
                  6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

# drop unwanted columns
features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']
# final train set
print(train)
