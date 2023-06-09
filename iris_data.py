from sklearn import datasets, metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris() #Loading the dataset
iris.keys()

dict_keys = (['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
iris = pd.DataFrame(
    data= np.c_[iris['data'], iris['target']],
    columns= iris['feature_names'] + ['target']
    )
iris.head(10)

species = []

for i in range(len(iris['target'])):
    if iris['target'][i] == 0:
        species.append("setosa")
    elif iris['target'][i] == 1:
        species.append('versicolor')
    else:
        species.append('virginica')

iris['species'] = species
iris.groupby('species').size()
iris.describe()

setosa = iris[iris.species == "setosa"]
versicolor = iris[iris.species=='versicolor']
virginica = iris[iris.species=='virginica']

# Droping the target and species since we only need the measurements
X = iris.drop(['target','species'], axis=1)
#print(X)

# converting into numpy array and assigning petal length and petal width
X = X.to_numpy()[:, (2,3)]
y = iris['target']

# Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5, random_state=42)

# Scaling the features for uniform evaluation
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

training_prediction = log_reg.predict(X_train)
#print(training_prediction)

test_prediction = log_reg.predict(X_test)
#print(test_prediction)

print("Precision, Recall, Confusion matrix, in training\n")

# Precision Recall scores
print(metrics.classification_report(y_train, training_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_train, training_prediction))

print("\nPrecision, Recall, Confusion matrix, in testing\n")

# Precision Recall scores
print(metrics.classification_report(y_test, test_prediction, digits=3))

# Confusion matrix
print(metrics.confusion_matrix(y_test, test_prediction))

def KN_neighbors():
    # Fitting KN_neighbors to the Training set
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Checking accuracy
    print('Accuracy output for KN_neighbors: ', accuracy_score(y_test, y_pred))

def naive_bayes():
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Checking accuracy
    print('Accuracy output for naive_bayes: ', accuracy_score(y_test, y_pred))

def decision_tree():
    # Fitting Decision Tree Classification to the Training set
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    # Checking accuracy
    print('Accuracy output for decision_tree: ', accuracy_score(y_test, y_pred))

def back_propagation():
    # Sets the size of hidden layers and number of iterations for feed-forward and back propagation
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    # Predicting the Test set results
    y_pred = mlp.predict(X_test)
    # Checking accuracy
    print('Accuracy output for back_propagation: ', accuracy_score(y_test, y_pred))

print()
KN_neighbors()
naive_bayes()
decision_tree()
back_propagation()
