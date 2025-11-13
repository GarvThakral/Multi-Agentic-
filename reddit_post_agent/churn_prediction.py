
import pandas as pd
import numpy as np
import kagglehub
from sklearn.preprocessing import OneHotEncoder

path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", path)

!ls $path

!mv $path/WA_Fn-UseC_-Telco-Customer-Churn.csv train.csv

data = pd.read_csv("train.csv")
data.columns

for column in data.columns:
  print(f"There are  {data[column].isnull().sum()} null values in the coilumn {column}")

columns_to_drop = ["customerID"]
data = data.drop(columns = columns_to_drop ,axis = 1)

# Gender preprocessing
data['gender'] = (data['gender'] == "Male").astype(int)

data['TotalCharges'] = data['TotalCharges'].apply(lambda x: 0.0 if x.strip() == '' else float(x))

from sklearn.preprocessing import StandardScaler
columns_to_scale = ["tenure" , "MonthlyCharges" , "TotalCharges"]
scalers = {}



for column in columns_to_scale:
  scaler = StandardScaler()
  column_scaler = scaler.fit(data[[column]])
  scalers[column] = column_scaler
  data[column] = column_scaler.transform(data[[column]])

columns_to_one_hot = ['MultipleLines',"InternetService" , "OnlineSecurity",
                      "OnlineBackup","DeviceProtection" , "TechSupport" ,
                      "StreamingTV", "StreamingMovies","Contract" ,
                      "PaperlessBilling","PaymentMethod"]

data = pd.get_dummies(data, columns=columns_to_one_hot, drop_first=False)

for column in data.columns:
  print(f"Unique values in {column} are {pd.unique(data[column])}")

for column in data.columns:
    unique_vals = data[column].dropna().unique()

    if set(unique_vals).issubset({0, 1}):
        data[column] = data[column].astype(int)

    elif data[column].nunique() == 2 and set(pd.Series(unique_vals).astype(str).str.lower()).issubset({'yes', 'no'}):
        data[column] = (data[column].str.lower() == 'yes').astype(int)

y_train = data["Churn"]
X_train = data.drop("Churn",axis = 1)
X_train.head(1)

X_test = X_train[-300:]
y_test = y_train[-300:]
X_train = X_train[:-300]
y_train = y_train[:-300]

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import tensorflow as tf

def simple_neural_net():
  inputs = tf.keras.layers.Input((41,))

  X = tf.keras.layers.Dense(200,'relu')(inputs)
  X = tf.keras.layers.BatchNormalization()(X)
  X = tf.keras.layers.Dense(200,'relu')(X)
  X = tf.keras.layers.BatchNormalization()(X)
  outputs = tf.keras.layers.Dense(1,'sigmoid')(X)

  model = tf.keras.Model(inputs = inputs , outputs = outputs)

  return model

models_to_test = ['LogisticRegression',"XGBoost","NeuralNet"]
model_results = {}
def model_trainer(model_name,X_train,y_train,X_test,y_test):
  if model_name == "LogisticRegression":
    model = LogisticRegression(max_iter=1000) # Increased max_iter to help with convergence
    trained_model = model.fit(X_train,y_train)
    result = trained_model.predict(X_test)
    score = accuracy_score(result,y_test)
    model_results[model_name] = score
  elif model_name == "XGBoost":
    model = XGBClassifier()
    model = model.fit(X_train,y_train)
    result = model.predict(X_test)
    score = accuracy_score(result,y_test)
    model_results[model_name] = score
  else:
    model = simple_neural_net()
    model.compile("adam","BinaryCrossentropy",metrics = ["accuracy"])
    history = model.fit(X_train,y=y_train,validation_data=[X_test,y_test],epochs = 4)
    score = history.history['val_accuracy'][-1]
    model_results[model_name] = score

def model_tester():
  for model in models_to_test:
    model_trainer(model,X_train_balanced,y_train_balanced,X_test,y_test)
  best_model = None
  max_score = 0
  for model_name,score in model_results.items():
    print(f"Model {model_name} accuracy = {score}")
    if score > max_score:
      max_score = score
      best_model = model_name

  print(f"The best model is {best_model}")
model_tester()