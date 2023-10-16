import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv")
data.head()
data.shape
data.info()
data.isnull().sum()
data.isnull().sum()/data.shape[0]*100
data.nunique()

sns.displot(data["Purchase"],color='r')
plt.title("Purchase Distribution")
plt.show()

sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()

data["Purchase"].skew()
data["Purchase"].kurtosis()
data["Purchase"].describe()
sns.countplot(data['Gender'])
plt.show()

data['Gender'].value_counts(normalize=True)*100
data.groupby("Gender").mean(numeric_only=True)["Purchase"]

plt.hist(data['Marital_Status'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Frequency')
plt.show()

data.groupby("Marital_Status").mean()["Purchase"]

data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
plt.title("Marital_Status and Purchase Analysis")
plt.show()

occupation_counts = data['Occupation'].value_counts()
plt.bar(occupation_counts.index, occupation_counts.values, color='skyblue', edgecolor='black')
plt.title('Occupation Distribution')
plt.xlabel('Occupation')
plt.ylabel('Frequency')
plt.show()

occup = pd.DataFrame(data.groupby("Occupation").mean()["Purchase"])
occup

occup.plot(kind='bar',figsize=(15,5))
plt.title("Occupation and Purchase Analysis")
plt.show()

sns.countplot(data['City_Category'])
plt.show()

data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
plt.title("City Category and Purchase Analysis")
plt.show()

sns.countplot(data['Stay_In_Current_City_Years'])
plt.show()

data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
plt.title("Stay_In_Current_City_Years and Purchase Analysis")
plt.show()

sns.countplot(data['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()

data.groupby("Age").mean()["Purchase"].plot(kind='bar')

data.groupby("Age").sum()['Purchase'].plot(kind="bar")
plt.title("Age and Purchase Analysis")
plt.show()

product_category_1_counts = data['Product_Category_1'].value_counts()
plt.bar(product_category_1_counts.index, product_category_1_counts.values, color='skyblue', edgecolor='black')
plt.title('Product Category 1 Distribution')
plt.xlabel('Product Category 1')
plt.ylabel('Frequency')
plt.show()

data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()

data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Analysis")
plt.show()

product_category_2_counts = data['Product_Category_2'].value_counts()
plt.bar(product_category_2_counts.index, product_category_2_counts.values, color='green', edgecolor='black')
plt.title('Product Category 2 Distribution')
plt.xlabel('Product Category 2')
plt.ylabel('Frequency')
plt.show()

product_category_3_counts = data['Product_Category_3'].value_counts()
plt.bar(product_category_3_counts.index, product_category_3_counts.values, color='blue', edgecolor='black')
plt.title('Product Category 3 Distribution')
plt.xlabel('Product Category 3')
plt.ylabel('Frequency')
plt.show()

data.corr()

sns.heatmap(data.corr(),annot=True)
plt.show()

data.columns
df = data.copy()
df.head()

# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace="4+",value="4")
#Dummy Variables:
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])

from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()

df['Gender'] = lr.fit_transform(df['Gender'])
df['Age'] = lr.fit_transform(df['Age'])
df['City_Category'] = lr.fit_transform(df['City_Category'])
df.head()

df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')

df.isnull().sum()

df.info()

df = df.drop(["User_ID","Product_ID"],axis=1)
X = df.drop("Purchase",axis=1)
y=df['Purchase']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.intercept_
lr.coef_
y_pred = lr.predict(X_test)

from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
mean_absolute_error(y_test, y_pred)

mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, y_pred)))

from sklearn.tree import DecisionTreeRegressor

# create a regressor object
regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X_train, y_train)
dt_y_pred = regressor.predict(X_test)
mean_absolute_error(y_test, dt_y_pred)
mean_squared_error(y_test, dt_y_pred)
r2_score(y_test, dt_y_pred)

from math import sqrt
print("RMSE of Decision Trees Regression Model is ",sqrt(mean_squared_error(y_test, dt_y_pred)))

from sklearn.ensemble import RandomForestRegressor

# create a regressor object
RFregressor = RandomForestRegressor(random_state = 0)

RFregressor.fit(X_train, y_train)

#C:\Users\Nantha\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
 # "10 in version 0.20 to 100 in 0.22.", FutureWarning)

rf_y_pred = RFregressor.predict(X_test)
mean_absolute_error(y_test, rf_y_pred)
mean_squared_error(y_test, rf_y_pred)
r2_score(y_test, rf_y_pred)

from math import sqrt
print("RMSE of Random Forest Regression Model is ",sqrt(mean_squared_error(y_test, rf_y_pred)))

from xgboost.sklearn import XGBRegressor
xbg_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)
xgb_y_pred = xgb_reg.predict(X_test)
mean_absolute_error(y_test, xgb_y_pred)
mean_squared_error(y_test, xgb_y_pred)
r2_score(y_test, xgb_y_pred)

from math import sqrt
print("RMSE of XGBoost Regression Model is ", sqrt(mean_squared_error(y_test, xgb_y_pred)))
