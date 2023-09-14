
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

#Read the data set
data = pd.read_csv("https://github.com/anthea97/GradientDescent/raw/main/winequality-red.csv",delimiter=";")

# Remove Null Values
data.dropna( inplace = True )

# Remove redundant Values
data.drop_duplicates()

# Observe the correlation matrix
correlation_matrix = data.corr()
sns.set(rc = {'figure.figsize':(15,10)})
sns.heatmap(correlation_matrix, annot=True,square=True)
plt.show()

"""From the correlation matrix, we can observe that the variables residual sugar, free sulphur dioxide and pH are weakly correlated with quality."""

# Drop output variable and weakly correlated predictors
X = data.drop(['quality',"free sulfur dioxide",'pH','residual sugar'],axis = 1)
print(X)

#Standardize x
sc=StandardScaler()
X = sc.fit_transform(X)
print(X)

#Split the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

"""
Part 1 - Custom Gradient Descent

"""

#Custom Gradient Descent Class
class customGD:
    def __init__(self, threshold):
        self.threshold = threshold

    def djdw(self,x,y,y_pred):
        res = 0
        for i in range(len(x)):
            res += x[i]*(y_pred[i]-y[i])
        return res/len(x)

    def djdb(self,y,y_pred):
        res = 0
        for i in range(len(y)):
            res+= y_pred[i]-y[i]
        return res/len(y)

    # returns y_predicted
    def predict(self,x,w,b):
        res = 0
        y_pred = []
        for i in range(len(x)):
            res = 0
            for j,a in enumerate(x[i]):
                res+= w[j]*a
            res+= b
            y_pred.append(res)
        return y_pred

    def error(self,y,y_pred):
        res = 0
        for i in range(len(y)):
            res+=(y_pred[i]-y[i])**2
        return res/len(y)

    def gradientDescent(self, x,y,iterations, learning_rate):
        y = y.to_numpy(dtype="float32")
        print(x)
        w = [0]*len(x[0])
        b = 0
        for _ in range(iterations):
            y_pred = self.predict(x,w,b)
            err = self.error(y,y_pred)
            if np.all(np.abs(learning_rate*self.djdw(x,y,y_pred)) <= self.threshold):
                break
            if np.all(np.abs(learning_rate*self.djdb(y,y_pred)) <= self.threshold):
                break
            w = w - learning_rate*self.djdw(x,y,y_pred)
            b = b - learning_rate*self.djdb(y,y_pred)
        return w,b

gd = customGD(0.001)

w,b = gd.gradientDescent(X_train,Y_train,10000,0.1)
print(w)
print(b)

y_pred = gd.predict(X_test,w,b)
print(mean_squared_error(Y_test,y_pred))



