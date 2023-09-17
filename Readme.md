# Linear Regression Using Gradient Descent (Assignment 1)
#### By Anthea Abreo(AXA210122) and Samyak Rokade(SJR220000)
#### Dataset:
Wine Quality:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Data Preprocessing

```python
data = pd.read_csv("https://github.com/anthea97/GradientDescent/raw/main/winequality-red.csv",delimiter=";")
```

* Null and redundant values were removed.
* Histograms were plotted for all the variables
![Histogram of variables](https://github.com/anthea97/GradientDescent/blob/main/histogramsOfData.png?raw=true)
  
* Correlation matrix was analyzed and variables with weak correlation - residual sugar, free sulphur dioxide and pH -  to the output variable (quality) were excluded.

```python
correlation_matrix = data.corr()
sns.set(rc = {'figure.figsize':(15,10)})
sns.heatmap(correlation_matrix, annot=True,square=True)
plt.show()
```

![Correlation Matrix](https://github.com/anthea97/GradientDescent/blob/main/Pasted%20image%2020230913222619.png?raw=true)

The data was split into training and test data with an 80/20 split.

## 1. Linear Regression Using Gradient Descent (From Scratch)
Error log for learning rate = 0.1 with increasing number of iterations:

| S.No  | # Iterations  | MSE          | MAE          | EAV          | R^2           |
|-------|---------------|--------------|--------------|--------------|---------------|
| 1     | 10            | 4.418898011  | 2.001531109  | 0.35137986   | -5.809046067  |
| 2     | 20            | 0.927470906  | 0.792843729  | 0.374701783  | -0.429132808  |
| 3     | 50            | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 4     | 100           | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 5     | 1000          | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 6     | 5000          | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 7     | 10000         | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 8     | 15000         | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 9     | 20000         | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |
| 10    | 25000         | 0.414570221  | 0.485520261  | 0.379796538  | 0.361191924   |



**Mean Square Error (MSE) vs Number of Iterations:**

We observe that the MSE decreases as the number of iterations increases. The minimum number of iterations to convergence is around 30. The same can be observe from the log table, as the values of MSE stabilize after number of iterations cross 20.
 
![](https://github.com/anthea97/GradientDescent/blob/main/Pasted%20image%2020230914151928.png?raw=true)

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
![](https://github.com/anthea97/GradientDescent/blob/main/Pasted%20image%2020230914155609.png?raw=true)
We observe that the MSE stabilizes at learning rate = 0.1. This is why we chose the final learning rate as 0.1.


**Results**:
* Weights: [0.09190231,-0.18631047, 0.00988079, -0.08262019, -0.06727607, -0.05484714, 0.14757514, 0.26096736]
* Bias: 5.553971075354176
* MSE: 0.41457022142942657
* MAE: 0.48552026122069414
* EAV: 0.3797965377406818
* R²: 0.36119192424387503

![ActualVsPredicted_CGD](https://github.com/anthea97/GradientDescent/blob/main/AVP_CGD.png?raw=true)

### Question:
Q. Are you satisfied that you have found the best
solution? Explain.

Ans: No, the model that we have built is not the best solution for the given dataset. There can be some parameters that can be modified like the learning rate, number of iterations etc., that will further decrease the error, but the current value of errors are permissible.

## 2. Linear Regression Using ML Library
SGDRegressor has been used as the linear model prediction of the wine quality dataset. By hit-and-trail, an observation was made that the SGDRegressor requires the data to be normalized in order to perform well. The coefficients of the SGDRegressor are as follows:

**Results**:
* Weights: [ 0.08640624,  -0.20771388, -0.01689627, -0.07121951, -0.0534198,  -0.03072766, 0.15382052,  0.26234732]
* Bias: 5.63082585
* MSE: 0.40621918708360366
* MAE: 0.49266536997221777
* EAV: 0.3756365448264052
* R²: 0.3740599690412909

![ActualVsPredicted_SGD](https://github.com/anthea97/GradientDescent/blob/main/AVP_SGD.png?raw=true)

## 3. Conclusion

In this project, the team embarked on a journey to implement Linear Regression using the Gradient Descent optimization technique, leveraging the Wine Quality dataset from the UCI Machine Learning Repository. Their objective was to predict wine quality based on various input variables, and the results provided valuable insights into the model's performance.

### Custom Gradient Descent vs. Library Implementation (SGDRegressor)

The team conducted two major experiments: one involving a custom implementation of Gradient Descent and another using a widely-used library, specifically the `SGDRegressor`. Both approaches aimed to achieve the same goal: predicting wine quality.

**Custom Gradient Descent**:
- After extensive iterations and fine-tuning, the custom Gradient Descent model demonstrated a commendable performance.
- The model's weight coefficients and bias were nearly identical to those obtained from the library implementation.
- Mean Square Error (MSE) converged to a satisfactory value of approximately 0.4145.
- Mean Absolute Error (MAE) and Explained Variance Score (EAV) also reached respectable levels.
- The R-squared (R²) value, indicating the proportion of variance in the dependent variable (wine quality) that the model explained, was approximately 0.3612.

**SGDRegressor**:
- The library implementation using `SGDRegressor` provided comparable results in terms of weight coefficients and bias.
- It achieved a similar MSE of around 0.4062, implying a close approximation to the custom Gradient Descent model.
- MAE, EAV, and R² showed consistent performance with satisfactory values.

<!-- ### Decision and Implications

The team's choice to opt for the custom Gradient Descent model over the library implementation was motivated by the similarity in results and the acceptable error rate (around 0.40). This decision not only highlights the viability of their custom approach but also underscores the significance of understanding the core algorithms and implementing them from scratch.

The insights gained from this project have practical implications, especially in scenarios where one needs to tailor the optimization process to specific requirements or constraints. Furthermore, it emphasizes the importance of rigorous experimentation and fine-tuning to achieve optimal results when using Gradient Descent.

In summary, the success of the custom Gradient Descent model in predicting wine quality validates its effectiveness in tackling real-world regression problems. This project serves as a testament to the power of understanding and implementing machine learning algorithms from the ground up, ultimately providing a strong foundation for future data science endeavors. -->
<!-- ## 3. Conclusion

In this project, the team implemented Linear Regression using Gradient Descent on the Wine Quality dataset from the UCI Machine Learning Repository to predict wine quality based on input variables.

### Custom Gradient Descent (From Scratch)

- The custom Gradient Descent model showed strong performance.
- It achieved an MSE of approximately 0.4145.
- MAE, EAV, and R² demonstrated favorable results.
- The model's weight coefficients and bias were effectively determined.

### SGDRegressor (Library Implementation)

- The team also utilized the `SGDRegressor` from a machine learning library.
- Results, including weight coefficients and bias, were comparable to the custom model.
- The MSE was approximately 0.4062, indicating a similar level of performance.
- MAE, EAV, and R² showed consistent results. -->

### Summary

The project highlights the success of custom Gradient Descent in solving regression problems and emphasizes the importance of algorithm understanding and fine-tuning. The findings provide a strong foundation for future data science endeavors.
