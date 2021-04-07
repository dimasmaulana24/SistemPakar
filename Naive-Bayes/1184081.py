from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
# importing the data from csv files
df1 = pd.read_csv(
    'tubes2_HeartDisease_train.csv', na_values=['?'])
df1.shape
df1.dropna(thresh=8, inplace=True)
df1.dropna(subset=['Column11', 'Column12', 'Column13'], how='all', inplace=True)

df1.head(10)
df1['Column5'].replace(0, 221, inplace=True)
df1['Column14'].replace(2, 1, inplace=True)
df1['Column14'].replace(3, 1, inplace=True)
df1['Column14'].replace(4, 1, inplace=True)
##########        Data Cleaning        ##########
# median to fill missing values
df1['Column4'].fillna(df1['Column4'].median(), inplace=True)
df1['Column5'].fillna(df1['Column5'].median(), inplace=True)
df1['Column6'].fillna(0, inplace=True)
df1['Column7'].fillna(df1['Column7'].median(), inplace=True)
df1['Column8'].fillna(df1['Column8'].median(), inplace=True)
df1['Column9'].fillna(0, inplace=True)
df1['Column12'].fillna(0, inplace=True)
df1['Column13'].fillna(3, inplace=True)
df1['Column11'].fillna(2, inplace=True)
df1['Column10'].fillna(df1['Column10'].median(), inplace=True)
df1.info()
df1[df1['Column11'].isnull()]
df1.isnull().sum()
X = df1[['Column3', 'Column9', 'Column11', 'Column12', 'Column13']]
y = df1['Column14']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# df1['Column11'].value_counts()
model = GaussianNB()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

plt.scatter(y_test, prediction)
print(classification_report(y_test, prediction))

print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
metrics.accuracy_score(prediction, y_test)