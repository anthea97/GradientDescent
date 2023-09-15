# Linear Regression Using Gradient Descent

#### Dataset:
Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Data Preprocessing
---
```python
data = pd.read_csv("https://github.com/anthea97/GradientDescent/raw/main/winequality-red.csv",delimiter=";")
```

* Null and redundant values were removed.
* Histograms were plotted for all the variables
![[Pasted image 20230915152816.png]]
* Plot of quality vs predictors:
  ![[Pasted image 20230915154059.png]]
  
* Correlation matrix was analyzed and variables with weak correlation - residual sugar, free sulphur dioxide and pH -  to the output variable (quality) were excluded.

```python
correlation_matrix = data.corr()
sns.set(rc = {'figure.figsize':(15,10)})
sns.heatmap(correlation_matrix, annot=True,square=True)
plt.show()
```

![[Pasted image 20230913222619.png]]

The data was split into training and test data with an 80/20 split.

## 1. Linear Regression Using Gradient Descent (From Scratch)
---
Error log for learning rate = 0.1 with increasing number of iterations:

| S.No 	| # Iterations 	| MSE         	| MAE         	| EAV         	| R^2          	|
|------	|--------------	|-------------	|-------------	|-------------	|--------------	|
| 1    	| 10           	| 4.418898011 	| 2.001531109 	| 0.35137986  	| -5.809046067 	|
| 2    	| 20           	| 0.927470906 	| 0.792843729 	| 0.374701783 	| -0.429132808 	|
| 3    	| 50           	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 4    	| 100          	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 5    	| 1000         	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 6    	| 5000         	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 7    	| 10000        	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 8    	| 15000        	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 9    	| 20000        	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|
| 10   	| 25000        	| 0.414570221 	| 0.485520261 	| 0.379796538 	| 0.361191924  	|

**Mean Square Error (MSE) vs Number of Iterations:**
![[Pasted image 20230914151928.png]]

We observe that the MSE decreases as the number of iterations increases. The minimum number of iterations to convergence is around 30. The same can be observe from the log table, as the values of MSE stabilize after number of iterations cross 20.


**Mean Square Error (MSE) vs Learning Rate:**
```python
def mseVsLR(X_train, Y_train, X_test, Y_test, y_pred):
  gd = customGD(0.001)
  for lr in np.arange(0.01, 0.5, 0.01):
    w,b = gd.gradientDescent(X_train,Y_train,50,lr)
    y_pred = gd.predict(X_test,w,b)
    mse = mean_squared_error(Y_test,y_pred)
    plt.scatter(lr, mse)

  plt.xlabel("learning rate")
  plt.ylabel("mse")
  plt.show()  

```

MSE was plotted for a range of learning rates between 0.01 and 0.5. 
![[Pasted image 20230914155609.png]]
We observe that the MSE stabilizes at learning rate = 0.1. This is why we chose the final learning rate as 0.1.

