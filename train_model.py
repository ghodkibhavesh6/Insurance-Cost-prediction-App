import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#load dataset
df = pd.read_csv('Insurance data.csv')

#convert catogarial value into number
encoder = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoder[col] = le

#select feature and label
X = df.drop("charges", axis=1)
y = df["charges"]

#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#select the algorithm and train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

#save the model
import pickle
with open("insurance_model.pkl", "wb") as f:
    pickle.dump(model, f)

#save columns of model
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model trained and saved successfully")
