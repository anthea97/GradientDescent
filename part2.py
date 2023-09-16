import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

data = pd.read_csv("https://github.com/anthea97/GradientDescent/raw/main/winequality-red.csv",delimiter=";")
y= data['quality']
X = data.drop(['quality',"free sulfur dioxide",'pH','residual sugar'],axis = 1)
sc=StandardScaler()
X=sc.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

# reg = LinearRegression().fit(X_train,Y_train)
reg = SGDRegressor().fit(X_train,Y_train)
y_pred = reg.predict(X_test)

print(reg.coef_)
print(mean_squared_error(Y_test,y_pred))
print("MSE:", mean_squared_error(Y_test,y_pred))
print("MAE:", mean_absolute_error(Y_test, y_pred))
print("EAV:", explained_variance_score(Y_test, y_pred))
print("R^2:", r2_score(Y_test,y_pred))

# preparing for plotting the graphs
axis = [i for i in range(len(y_pred))]
ny_pred = [round(i) for i in y_pred]

# Plotting the actual results vs the predicted results
plt.figure()
plt.title(" Actual value vs Predicted value (SGDRegressor)")
plt.scatter(axis,Y_test,color="red", label="Actual value")
plt.scatter(axis,ny_pred,color="blue",label = "Predicted value")
plt.xlabel("Data-points")
plt.ylabel("Quality of wine")
plt.legend()
plt.show()