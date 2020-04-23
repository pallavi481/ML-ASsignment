import numpy as np 
import pandas as pd 
df=pd.read_csv("C:\\Users\\Hp\\Desktop\\Pallavi_flight\\pa_data.csv")

from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
X=df['DAY_OF_WEEK'].values.reshape(-1,1)
Y=df['DEST_AIRPORT_ID']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
p1 = regressor.predict([[44]])
p1[0]
