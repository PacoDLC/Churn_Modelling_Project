# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 19:51:10 2024

@author: Francisco De La Cruz
"""

""" <<<<<<<<<<<<<<<<<<<<<< ARTIFITIAL NEURAL NETWORK >>>>>>>>>>>>>>>>>>>>>> """

""" 
DESCRIPTION: This Python script uses an ANN model on a dataset containing 
characteristics of different customers of a dummy bank, who have decided to
leave or stay with the bank. The purpose of this model is to try to predict on
a test set whether certain customers will leave their account at the bank.
"""

import numpy as np 
import pandas as pd
import tensorflow as tf

tf.__version__

# Part 1 - Data Preprocessing

# Import Dataset

dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical features

# Column "Gender"
from sklearn.preprocessing import LabelEncoder

X[:, 2] = LabelEncoder().fit_transform(X[:, 2])

# Column "Geography"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [("encoder", OneHotEncoder(), [1])], \
                       remainder = "passthrough")
X = np.array(ct.fit_transform(X))

# Split the Dataset into Training and Testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, \
                                                    random_state = 0)
    
# Scaling features

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2: Building the ANN Model

# Initializating ANN Model

ann = tf.keras.models.Sequential()

# Add the input layer and the first hidden layer

ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = "uniform", \
                              activation = "relu", input_dim = 12))
    
# Add second hidden layer

ann.add(tf.keras.layers.Dense(units = 6, kernel_initializer = "uniform", \
                              activation = "relu"))

# Add output layer    

ann.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid"))
    
# Part 3: Training the ANN Model

# Compile the ANN Model

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', \
            metrics = ['accuracy'])
    
# Fit the ANN Model with the Training set

ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predict the Testing set

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), \
                      y_test.reshape(len(y_test),1)), 1))
    
# Making Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

"""
Task: Use our ANN Model to predict whether the customer with the featires 
bellow will leave the bank

CreditScore: 600
Geography: France
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
NumOfProducts: 2
HasCrCard: Yes
IsActiveMember: Yes
EstimatedSalary: 50000

""" 

customer = [1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]

print(ann.predict(sc.transform([customer])))

print(ann.predict(sc.transform([customer])) > 0.5)