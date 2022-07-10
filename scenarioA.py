import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn import set_config

#######################################################
## Functions that will be used to generate models

def quickFit(modelName, model, X, y):
    """Fits a model to a given dataset and displays accuracy, precision score and recall score.
    The function supposes that a preprocessor has already been created."""

    global preprocessor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    le = LabelEncoder()
    le.fit_transform(y_train)
    le.transform(y_test)

    model = Pipeline(steps=[('Preprocessing', preprocessor), (modelName, model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('\n----- ' + modelName + ' -----')
    print('Accuracy score: ' + str(accuracy_score(y_test, y_pred)))
    print('Precision score: ' + str(precision_score(y_test, y_pred)))
    print('Recall score: ' + str(recall_score(y_test, y_pred)))

    print("----- Confusion Matrix -----")
    print(confusion_matrix(y_test, y_pred))

#####################################################
## Importing and cleaning data

scenarioA = pd.read_csv('csvs/scenarioA.csv')

scenarioA.replace([np.inf, -np.inf], np.nan, inplace=True)
scenarioA.dropna(inplace=True)

## Preprocessor for data

p1 = ('OneHotEncoder', OneHotEncoder(), [' Protocol'])
p2 = ('StdScaler', StandardScaler(), [' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean'])
# p3 = ('Label Encoder', LabelEncoder(), ['label'])
preprocessor = ColumnTransformer([p1, p2])


#####################################################
# Model generation

useCols = [' Protocol', ' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean']

quickFit('Random Forest Classifier', RandomForestClassifier(random_state=0), scenarioA[useCols], scenarioA['label'].replace({'TOR': 1, 'nonTOR': 0}))
quickFit('Logistic Regression', LogisticRegression(random_state=0), scenarioA[useCols], scenarioA['label'].replace({'TOR': 1, 'nonTOR': 0}))
quickFit('K-Neighbors Classifier', KNeighborsClassifier(n_neighbors=5), scenarioA[useCols], scenarioA['label'].replace({'TOR': 1, 'nonTOR': 0}))
