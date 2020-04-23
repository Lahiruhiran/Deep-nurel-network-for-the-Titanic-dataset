import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle


test = pd.read_csv("test.csv")

test['Sex'] = test['Sex'].map({'male': 0, 'female': 1}).astype(int)
median = test['Age'].median()
test["Age"].fillna(median, inplace=True)

test['Embarked'].fillna("S", inplace=True)
test['Embarked'] = test['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2}).astype(int)

# define embarked ports for 3 categories


test.drop('Name', axis=1, inplace=True)
# split into input (X) and output (y) variables

feature_names = ['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']
X = test[feature_names]
scaler = MinMaxScaler()
X_test = scaler.fit_transform(X)


filename = 'finalized_model.h5'

loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test)
# print(result)
# make probability predictions with the model
predictions = loaded_model.predict(X_test)

# round predictions
# rounded = [round(X_test[0]) for x in predictions]

# make class predictions with the model
predictions = loaded_model.predict_classes(X_test)

print(predictions)
# summarize the first 5 cases
# for i in range(5):
#     print('%s => %d (expected %d)' %
#           (X_test[i].tolist(), predictions[i], y_test[i]))
