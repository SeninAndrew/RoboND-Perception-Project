#!/usr/bin/env python

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import accuracy_score

diabetes = datasets.load_diabetes()
X = diabetes.data[:400]
y = diabetes.target[:400]
cv = KFold(n_splits=20)
lasso = linear_model.Lasso()
y_pred = cross_val_predict(lasso, X, y, cv=cv)

print("y = ", y)
print("y_pred = ", y_pred)

accuracy = accuracy_score(y_pred.astype(int), y.astype(int))

print(accuracy)
