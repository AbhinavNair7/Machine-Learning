import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

contents = np.loadtxt(fname='ex2data2.txt',delimiter=',')

# Data converted to Polynomial
X = contents[:,0:2]
poly = PolynomialFeatures()
X_new = poly.fit_transform(X)
y = contents[:,2]

# Best regularisation loop
lmbda = 0.3
best_acc = 0
while lmbda > 0:
    model = LogisticRegression(max_iter=1000,C = 1/lmbda)
    fit = model.fit(X_new,y)
    acc = model.score(X_new,y)
    if acc > best_acc:
        best_acc = acc
        best_lambda = lmbda
    lmbda -= 0.01
print(best_acc,best_lambda)

# Logistic model fit
model = LogisticRegression(max_iter=1000,C = 1/best_lambda)
model.fit(X_new,y)
coef = model.coef_
intercept = model.intercept_

# Predictions from model
initial_y_pred = np.sum(np.multiply(X_new,coef),axis=1)+intercept
y_pred = model.predict(X_new)
pos_X_pred = X[y_pred==1]
true_pos_X = X[y==1]

# Plot predicted positive X1,X2 over true positive and negative
plt.figure(1)
plt.scatter(X[:,0],X[:,1],color='red',label='True Negatives')
plt.scatter(true_pos_X[:,0],true_pos_X[:,1],color='green',label='True Positives')
plt.scatter(pos_X_pred[:,0],pos_X_pred[:,1],marker='x',color='black',label='Predicted Positives')
plt.legend()
plt.show()

#Plot y predictions = 0 against true positive and negative for X1
plt.figure(2)
plt.scatter(X[:,0],initial_y_pred,color='red',label='True Negatives')
plt.scatter(true_pos_X[:,0],initial_y_pred[y==1],color='green',label='True Positives')
plt.plot(X[:,0],np.zeros(np.shape(X)[0]),label='Y prediction = 0')
plt.legend()
plt.show()
