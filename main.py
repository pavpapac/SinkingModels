# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import interpret as inter
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import set_visualize_provider
from interpret.provider import InlineProvider
from interpret import show
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer


def read_csv_path(path):
    # import data from csv file
    data_df = pd.read_csv(path)

    return data_df

def return_labels(df):
    #return column labels
    return list(df.columns)


def encode_categorical(data_df, list_columns):
    #encode categorical data, so that we can use them together with numerical
    categorical_data = data_df[list_columns].values
    enc = OrdinalEncoder()
    categorical_data_encoded = enc.fit_transform(categorical_data)
    categorical_data_encoded_df = pd.DataFrame(categorical_data_encoded, columns=['Sex', 'Embarked'])

    return categorical_data_encoded_df

def select_numerical(data_df, list_columns):
    # select the numerical fields to be used in training. Some may be irrelevant
        return data_df[list_columns]


def create_features_labels(num_df, cat_df, y_label):
    # merge now numerical and encoded categorical data in a dataset that can be sued for training
    X_df = pd.concat([num_df, cat_df], axis=1)
    y_df = data_df[y_label]

    return X_df, y_df

def append_existsNaN(X_df):
    # if NaN exists append a boolean label in the features. This can stay as a feature column as it says something about
    # the data
    exists_NaN_df = pd.DataFrame(np.int16(X_df.isna().sum(axis=1) == True), columns=['exists_NaN'])
    X_df = pd.concat([X_df, exists_NaN_df], axis=1)

    return X_df


def impute_with_KNN(X_df, n_neighborhs=5):
    # Replace Nan values with an estimation using K nearest neighbors algorithm. This method works better than
    # replacing with the median or average or remving completely the data
    imputer = KNNImputer(n_neighbors=n_neighborhs, weights="uniform")
    X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=np.array(X_df.columns))

    return X_df


def create_train_test_datasets(X_df, y_df, test_size=0.20):
    #Now create the final traina nd test datasets with a pre-determined ratio
    X = np.array(X_df)
    y = np.array(y_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    path = './Data/titanic/train.csv'
    data_df = read_csv_path(path)
    cat_df = encode_categorical(data_df, ['Sex', 'Embarked'])
    num_df = select_numerical(data_df, ['Age', 'SibSp', 'Pclass', 'Parch', 'Fare'])
    X_df, y_df = create_features_labels(num_df, cat_df, 'Survived')
    X_df = append_existsNaN(X_df)
    X_df = impute_with_KNN(X_df)
    X_train, X_test, y_train, y_test = create_train_test_datasets(X_df, y_df)
    print(X_train, X_test, y_train, y_test)
