import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from xgboost.sklearn import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.preprocessing import LabelEncoder

<<<<<<< HEAD
=======
data = pd.read_csv("customer.csv")
data.head()
data.shape
data.info()
data.isnull().sum()
data.isnull().sum()/data.shape[0]*100
data.nunique()
>>>>>>> e257d4e91b25e4e74b40efd3d2476b05bd623bbd

class BlackFridaySalesAnalysis:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def display_basic_info(self):
        print(self.data.head())
        print(self.data.shape)
        print(self.data.info())
        print(self.data.isnull().sum())
        print(self.data.isnull().sum() / self.data.shape[0] * 100)
        print(self.data.nunique())

    def visualize_purchase_distribution(self):
        sns.displot(self.data["Purchase"], color='r')
        plt.title("Purchase Distribution")
        plt.show()

    def visualize_purchase_boxplot(self):
        sns.boxplot(self.data["Purchase"])
        plt.title("Boxplot of Purchase")
        plt.show()

    # Add more visualization functions as needed
    def visualize_gender_distribution(self):
        self.data["Purchase"].skew()
        self.data["Purchase"].kurtosis()
        self.data["Purchase"].describe()
        sns.countplot(self.data['Gender'])
        plt.show()

    # Shows Marital Status Distribution
    def visualize_marital_status(self):
        self.data['Gender'].value_counts(normalize=True) * 100
        self.data.groupby("Gender").mean(numeric_only=True)["Purchase"]

        sns.countplot(self.data['Marital_Status'])
        plt.show()

    def visualize_marital_and_purchase(self):
        self.data.groupby("Marital_Status").mean()["Purchase"]

        self.data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
        plt.title("Marital_Status and Purchase Analysis")
        plt.show()

    def visualize_occupation_distribution(self):
        plt.figure(figsize=(18, 5))
        sns.countplot(self.data['Occupation'])
        plt.show()

    def visualize_occupation_purchase(self):
        occup = pd.DataFrame(self.data.groupby("Occupation").mean()["Purchase"])
        occup

        occup.plot(kind='bar', figsize=(15, 5))
        plt.title("Occupation and Purchase Analysis")
        plt.show()

    def visualize_city_category(self):
        sns.countplot(self.data['City_Category'])
        plt.show()

    def visualize_city_and_purchase(self):
        self.data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
        plt.title("City Category and Purchase Analysis")
        plt.show()

    def visualize_stay_in_years(self):
        sns.countplot(self.data['Stay_In_Current_City_Years'])
        plt.show()

    def visualize_stay_in_and_purchase(self):
        self.data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
        plt.title("Stay_In_Current_City_Years and Purchase Analysis")
        plt.show()

    def visualize_age_distribution(self):
        sns.countplot(self.data['Age'])
        plt.title('Distribution of Age')
        plt.xlabel('Different Categories of Age')
        plt.show()

    def visualize_age_and_purchase(self):
        self.data.groupby("Age").mean()["Purchase"].plot(kind='bar')

        self.data.groupby("Age").sum()['Purchase'].plot(kind="bar")
        plt.title("Age and Purchase Analysis")
        plt.show()

    def visualize_product_one_analysis(self):
        plt.figure(figsize=(18, 5))
        sns.countplot(self.data['Product_Category_1'])
        plt.show()

        self.data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_1 and Purchase Mean Analysis")
        plt.show()

        self.data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_1 and Purchase Analysis")
        plt.show()

    def visualize_product_two_analysis(self):
        plt.figure(figsize=(18, 5))
        sns.countplot(self.data['Product_Category_2'])
        plt.show()

        self.data.groupby('Product_Category_2').mean()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_2 and Purchase Mean Analysis")
        plt.show()

        self.data.groupby('Product_Category_2').sum()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_2 and Purchase Analysis")
        plt.show()

    def visualize_product_three_analysis(self):
        plt.figure(figsize=(18, 5))
        sns.countplot(self.data['Product_Category_3'])
        plt.show()

        self.data.groupby('Product_Category_3').mean()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_3 and Purchase Mean Analysis")
        plt.show()

        self.data.groupby('Product_Category_3').sum()['Purchase'].plot(kind='bar', figsize=(18, 5))
        plt.title("Product_Category_3 and Purchase Analysis")
        plt.show()

    def visualize_heatmap(self):
        self.data.corr()
        sns.heatmap(self.data.corr(), annot=True)
        plt.show()

    def preprocess_data(self):
        # Dropping User_ID and Product_ID columns
        self.data = self.data.drop(["User_ID", "Product_ID"], axis=1)

        # Handling missing values in Product_Category_2 and Product_Category_3
        self.data['Product_Category_2'].fillna(0, inplace=True)
        self.data['Product_Category_3'].fillna(0, inplace=True)

        # Encoding categorical variables
        lr = LabelEncoder()
        self.data = pd.get_dummies(self.data, columns=['Stay_In_Current_City_Years'])
        self.data['Gender'] = lr.fit_transform(self.data['Gender'])
        self.data['Age'] = lr.fit_transform(self.data['Age'])
        self.data['City_Category'] = lr.fit_transform(self.data['City_Category'])

        # Separating features and target variable
        self.X = self.data.drop("Purchase", axis=1)
        self.y = self.data['Purchase']

        # Splitting the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                random_state=123)

    def train_linear_regression(self):
        # Add code for training linear regression model here
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        return lr

    def evaluate_linear_regression_model(self, model, X_test, y_test):
        # Add code for evaluating linear regression model here
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2}

    # Similar functions for other regression models (Decision Tree, Random Forest, XGBoost)

    def train_decision_tree_regression(self):
        dt = DecisionTreeRegressor(random_state = 0)
        dt.fit(self.X_train, self.y_train)
        return dt

    def evaluate_decision_tree_regression_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2}

    def train_random_forest_regression(self):
        RFregressor = RandomForestRegressor(random_state = 0)
        RFregressor.fit(self.X_train, self.y_train)
        return RFregressor

    def evaluate_random_forest_regression_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2}

    def train_XGBoost_regression(self):
        XGBregressor = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)
        XGBregressor.fit(self.X_train, self.y_train)
        return XGBregressor

    def evaluate_XGBoost_regression_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R-squared': r2}

def main():
    data_file = "customer.csv"
    analysis = BlackFridaySalesAnalysis(data_file)

    #analysis.display_basic_info()
    analysis.visualize_purchase_distribution()
    analysis.visualize_purchase_boxplot()
    analysis.visualize_gender_distribution()
    #analysis.visualize_marital_status()
    analysis.visualize_marital_and_purchase()
    #analysis.visualize_occupation_distribution()
    analysis.visualize_occupation_purchase()
    analysis.visualize_city_category()
    analysis.visualize_city_and_purchase()
    analysis.visualize_stay_in_years()
    analysis.visualize_stay_in_and_purchase()
    analysis.visualize_age_distribution()
    analysis.visualize_age_and_purchase()
    #analysis.visualize_product_one_analysis()
    #analysis.visualize_product_two_analysis()
    #analysis.visualize_product_three_analysis()
    analysis.visualize_heatmap()

    # Add more analysis and modeling steps as needed

    # Preprocess the data
    analysis.preprocess_data()

    # Train a Linear Regression model
    lr_model = analysis.train_linear_regression()

    # Evaluate the model
    evaluation_metrics = analysis.evaluate_linear_regression_model(lr_model, analysis.X_test, analysis.y_test)
    print("Linear Regression Model Evaluation:")
    print(evaluation_metrics)

    dt_model = analysis.train_decision_tree_regression()

    dt_evaluation_metrics = analysis.evaluate_decision_tree_regression_model(dt_model, analysis.X_test, analysis.y_test)
    print("Decision Tree Regression Model Evaluation:")
    print(dt_evaluation_metrics)

    rf_model = analysis.train_random_forest_regression()

    rf_evaluation_metrics = analysis.evaluate_random_forest_regression_model(rf_model, analysis.X_test, analysis.y_test)
    print("Random Forest Regression Model Evaluation:")
    print(rf_evaluation_metrics)

    xgb_model = analysis.train_XGBoost_regression()

    xgb_evaluation_metrics = analysis.evaluate_XGBoost_regression_model(xgb_model, analysis.X_test, analysis.y_test)
    print("XGBoost Regression Model Evaluation:")
    print(xgb_evaluation_metrics)


if __name__ == "__main__":
    main()