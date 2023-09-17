import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
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

fig = plt.subplots(figsize=(10,8))
ax = sns.heatmap(data.corr(), annot=True, linewidths=2)

"""From the correlation matrix, we can observe that the variables residual sugar, free sulphur dioxide and pH are weakly correlated with quality."""

# Drop output variable and weakly correlated predictors
y = data['quality']
X = data.drop(['quality',"free sulfur dioxide",'pH','residual sugar'],axis = 1)
print(X)

#Standardize x
sc=StandardScaler()
X = sc.fit_transform(X)
print(X)

#Split the dataset
X=sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

"""
Part 1 - Custom Gradient Descent

"""

#Custom Gradient Descent Class
class customGD:
    def __init__(self, threshold,max_iterations,learning_rate):
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

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

    def gradientDescent(self, x,y):
        y = y.to_numpy(dtype="float32")
        print(x)
        w = [0]*len(x[0])
        b = 0
        for _ in range(self.max_iterations):
            y_pred = self.predict(x,w,b)
            # err = self.error(y,y_pred)
            # if the value of difference is less than the threshold the algo stops.
            if np.all(np.abs(self.learning_rate*self.djdw(x,y,y_pred)) <= self.threshold):
                break
            if np.all(np.abs(self.learning_rate*self.djdb(y,y_pred)) <= self.threshold):
                break
            w = w - self.learning_rate*self.djdw(x,y,y_pred)
            b = b - self.learning_rate*self.djdb(y,y_pred)
        return w,b

threshold = 0.001
max_iterations = 50
learning_rate = 0.1
gd = customGD(threshold,max_iterations,learning_rate)
w,b = gd.gradientDescent(X_train,Y_train)
print("weights = ",w)
print("bias = ",b)

y_pred = gd.predict(X_test,w,b)
print("MSE:", mean_squared_error(Y_test,y_pred))
print("MAE:", mean_absolute_error(Y_test, y_pred))
print("EAV:", explained_variance_score(Y_test, y_pred))
print("R^2:", r2_score(Y_test,y_pred))

# preparing for plotting the graphs
axis = [i for i in range(len(y_pred))]
ny_pred = [round(i) for i in y_pred]

# Plotting the actual results vs the predicted results
plt.figure()
plt.title("Actual value vs Predicted value (Custom Gradient Descent)")
plt.scatter(axis,Y_test,color="red", label="Actual value")
plt.scatter(axis,ny_pred,color="blue",label = "Predicted value")
plt.xlabel("Data-points")
plt.ylabel("Quality of wine")
plt.legend()
plt.show()