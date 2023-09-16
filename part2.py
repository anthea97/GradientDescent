import pandas as pd
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("wineDataset/winequality-red.csv",delimiter=";")
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