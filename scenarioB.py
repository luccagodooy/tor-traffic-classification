import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

###################

def quickFit(modelName, model, X, y):
    global preprocessor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    le = LabelEncoder()
    le.fit_transform(y_train)
    le.transform(y_test)

    model = Pipeline(steps=[('Preprocessor', preprocessor), (modelName, model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n-----" + modelName + ' -----')
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


###################

scenarioB = pd.read_csv('csvs/scenarioB.csv')

p1 = ('OneHotEncoder', OneHotEncoder(), [' Protocol'])
p2 = ('StdScaler', StandardScaler(), [' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean'])
preprocessor = ColumnTransformer([p1, p2])

useCols = [' Protocol', ' Flow Duration', ' Flow Bytes/s', ' Flow Packets/s', ' Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Active Mean', 'Idle Mean']

###################
quickFit('Random Forest Classifier', RandomForestClassifier(random_state=0), scenarioB[useCols], scenarioB['label'])
quickFit('Decision Tree Classifier', DecisionTreeClassifier(random_state=0), scenarioB[useCols], scenarioB['label'])
quickFit('K-Neighbor Classifier', KNeighborsClassifier(), scenarioB[useCols], scenarioB['label'])
