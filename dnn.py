
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import joblib
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
# load the dataset

# load the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train_test_data = [train, test]

sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
median = train['Age'].median()
train["Age"].fillna(median, inplace=True)

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# define embarked ports for 3 categories
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train.drop('Name', axis=1, inplace=True)
# split into input (X) and output (y) variables


# X = train[:, 1:8]
# y = train[:, 8]

feature_names = ['Age', 'Sex', 'SibSp', 'Parch', 'Pclass']
X = train[feature_names]
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# define the keras model
model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy',
              optimizer='SGD', metrics=['accuracy'])

# fit the keras model on the dataset

earlyStopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint(
    '.finalized_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(
    monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


model.fit(X_train, y_train, epochs=2000, batch_size=5, callbacks=[
          earlyStopping, mcp_save, reduce_lr_loss], validation_split=0.25)


# evaluate the keras model
_, accuracy = model.evaluate(X_test, y_test)
print('Accuracy: %.2f' % (accuracy * 100))


# Save the trained model as a pickle string.
# saved_model = pickle.dumps(model)

# Load the pickled model
# knn_from_pickle = pickle.loads(saved_model)

# filename = 'finalized_model.h5'
# joblib.dump(model, filename)
