# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
        return theta
data=pd.read_csv("C:\\Users\\Pandiyan\\Downloads\\50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
 <img width="791" height="404" alt="image" src="https://github.com/user-attachments/assets/903bdf1f-8326-42cc-a499-bad40c6f1979" />

<img width="420" height="389" alt="image" src="https://github.com/user-attachments/assets/f96c0d72-5bd3-4178-a42a-ee1e2ea42774" />

<img width="468" height="391" alt="image" src="https://github.com/user-attachments/assets/cdcd677a-f05c-4bc7-85be-b5c6a8059bb6" />

<img width="639" height="390" alt="image" src="https://github.com/user-attachments/assets/835916d0-2d02-4ea1-aa79-103022f6df46" />

<img width="653" height="402" alt="image" src="https://github.com/user-attachments/assets/662cd73e-7c8e-480c-8998-eaa7b6bbd627" />
<img width="631" height="396" alt="image" src="https://github.com/user-attachments/assets/834d34e3-d123-476a-bc7f-90118c2a7882" />

<img width="549" height="372" alt="image" src="https://github.com/user-attachments/assets/0e230c27-4837-4505-8eed-c393cc59e088" />






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
