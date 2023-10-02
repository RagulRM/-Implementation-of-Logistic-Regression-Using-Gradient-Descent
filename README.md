# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Ragul R
RegisterNumber:  212222100040

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/
```

## Output:

![Screenshot (292)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/0bf3f4f2-0e87-44aa-bfad-5b0bb23a5ef2)


![Screenshot (292) 1](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/ede9d59a-e3f5-4e75-b872-73fa48e269ba)


![Screenshot (263)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/3a2b9e11-8db0-4269-9033-b63a850796b9)

![Screenshot (264)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/40800023-594f-4b2a-bdab-bee74f6738cd)

![Screenshot (265)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/251f8129-4db4-49af-8e7a-56424b7054e6)

![Screenshot (265) 1](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/eca4766c-2f32-4cc0-8416-6e5d5f5457c3)

![Screenshot (293)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/a4315f6e-486f-4c52-8387-3428393f5d06)


![Screenshot (267)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/55560375-278d-4540-98e4-f4b7560781f8)

![Screenshot (294)](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/3f3fe9bd-2d16-4693-8bd9-39c196ee8dfd)


![Screenshot (294) 1](https://github.com/Anusharonselva/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119405600/21f754eb-8d33-4a6d-9e5a-884c804ee36f)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

