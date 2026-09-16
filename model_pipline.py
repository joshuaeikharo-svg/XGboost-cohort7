import pandas as pd
import xgboost
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# Load the dataset
dataset = pd.read_csv("data/insurance.csv")

# Handling categorical variables
dataset.drop(['CustomerId', 'Surname'], axis = 1, inplace=True)
dataset['Geography'].unique()
geography_dummies = pd.get_dummies(['Geography'], drop_first=True)
dataset = pd.concat([geography_dummies, dataset], axis=1)
dataset.drop(['Geography'], axis=1, inplace=True)

# gender
dataset['Gender'] = dataset['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Creating the Training Set and the Test Set
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Build and train the model
model = xgboost.XGBClassifier(max_depth=8, learning_rate=0.3, n_estimators=100)

model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
joblib.dump(model, 'model/model.pkl')
print("Model saved as 'model.pkl'")

#if __name__ == "__main__":
    #print(dataset)

