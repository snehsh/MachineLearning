import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

df1 = df[['Mileage', 'Price']]
bins = np.arange(0, 50000, 10000)
groups = df1.groupby(pd.cut(df1['Mileage'], bins)).mean()
groups['Price'].plot.line()

scale = StandardScaler()

X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']

X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].values)
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

y.groupby(df.Doors).mean()

scaled = scale.transform([[45000, 8, 4]])
scaled = np.insert(scaled[0], 0, 1)
predicted = est.predict(scaled)
print(predicted)
