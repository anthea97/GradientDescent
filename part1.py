# Linear Regression using Gradient Descent
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("wineDataset/winequality-red.csv",delimiter=";")
y= data['quality']
X = data.drop(['quality',"free sulfur dioxide",'pH','residual sugar'],axis = 1)
sc=StandardScaler()
X=sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)


def djdw(x,y,y_pred):
    res = 0
    for i in range(len(x)):
        res += x[i]*(y_pred[i]-y[i])
    return res/len(x)

def djdb(y,y_pred):
    res = 0
    for i in range(len(y)):
        res+= y_pred[i]-y[i]
    return res/len(y)

# returns y_predicted
def predict(x,w,b):
    res = 0
    y_pred = []
    for i in range(len(x)):
        res = 0
        for j,a in enumerate(x[i]):
            res+= w[j]*a
        res+= b
        y_pred.append(res)
    return y_pred

def error(y,y_pred):
    res = 0
    for i in range(len(y)):
        res+=(y_pred[i]-y[i])**2
    return res/len(y)
# comment or delete before submitting
def gradientDescentAlpha(x,y,iterations, learning_rate,threshold = 0.001):
    y = y.to_numpy(dtype="float32")
    w = [0]*len(x[0])
    b = 0 
    for _ in range(iterations):
        y_pred = predict(x,w,b)
        err = error(y,y_pred)
        if np.all(np.abs(learning_rate*djdw(x,y,y_pred)) <= threshold):
            break
        if np.all(np.abs(learning_rate*djdb(y,y_pred)) <= threshold):
            break
        w = w - learning_rate*djdw(x,y,y_pred)
        b = b - learning_rate*djdb(y,y_pred)
    plt.scatter(learning_rate,b,color="blue")
for i in np.arange(0.01,0.5,0.01):
    gradientDescentAlpha(X_train,Y_train,10000,i,0.001)
plt.show()
    

def gradientDescent(x,y,iterations, learning_rate,threshold = 0.001):
    y = y.to_numpy(dtype="float32")
    print(x)
    w = [0]*len(x[0])
    b = 0 
    print(threshold)
    for _ in range(iterations):
        y_pred = predict(x,w,b)
        err = error(y,y_pred)
        if np.all(np.abs(learning_rate*djdw(x,y,y_pred)) <= threshold):
            break
        if np.all(np.abs(learning_rate*djdb(y,y_pred)) <= threshold):
            break
        w = w - learning_rate*djdw(x,y,y_pred)
        b = b - learning_rate*djdb(y,y_pred)
    return w,b
    
# Uncomment before submitting
w,b = gradientDescent(X_train,Y_train,50,0.1)
y_pred = predict(X_test,w,b)
print(mean_squared_error(Y_test,y_pred))
print(w,b)
