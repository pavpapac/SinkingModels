import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


class ETL:

    def __init__(self, path, list_cat, list_num, target_label):
        self.path = path
        self.list_cat = list_cat
        self.list_num = list_num
        self.target_label = target_label
        self.data_df = pd.DataFrame()
        self.X_df = pd.DataFrame()
        self.y_df = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.feature_names = np.array([])
        self.columns = np.array([])

    def pipeline(self):
        # High-level function calling all steps sequentially
        self.import_data_df()
        self.clean_data_pipeline()
        self.create_train_test_datasets()

    def import_data_df(self):
        # import data from csv file
        self.data_df = pd.read_csv(str(self.path))
        self.columns = np.array(self.data_df.columns)

    def clean_data_pipeline(self):
        # Pipeline of actions to clean up and prepare datasets prior to splitting in train and test
        cat_df = self._encode_categorical()
        num_df = self._select_numerical()
        X_df, self.y_df = self._create_features_labels(num_df, cat_df, self.target_label)
        X_df = self._append_existsNaN(X_df)
        self.X_df = self._impute_with_KNN(X_df)
        self.feature_names = self._feature_names(X_df)

    def create_train_test_datasets(self, test_size=0.20):
        # Now create the final train and test datasets with a pre-determined ratio
        X = np.array(self.X_df)
        y = np.array(self.y_df)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size)

    def _encode_categorical(self):
        # encode categorical data, so that we can use them together with numerical

        categorical_data = self.data_df[self.list_cat].values
        enc = OrdinalEncoder()
        categorical_data_encoded = enc.fit_transform(categorical_data)
        categorical_data_encoded_df = pd.DataFrame(categorical_data_encoded, columns=self.list_cat)

        return categorical_data_encoded_df

    def _select_numerical(self):
        # select the numerical fields to be used in training. Some may be irrelevant
        return self.data_df[self.list_num]

    def _create_features_labels(self, num_df, cat_df, y_label):
        # merge now numerical and encoded categorical data in a dataset that can be sued for training
        X_df = pd.concat([num_df, cat_df], axis=1)
        y_df = self.data_df[y_label]

        return X_df, y_df

    def _append_existsNaN(self, X_df):
        # if NaN exists append a boolean label in the features. This can stay as a feature column
        # as it says something about the data
        exists_NaN_df = pd.DataFrame(np.int16(X_df.isna().sum(axis=1) == True), columns=['exists_NaN'])
        X_df = pd.concat([X_df, exists_NaN_df], axis=1)

        return X_df

    def _impute_with_KNN(self, X_df, n_neighborhs=5):
        # Replace Nan values with an estimation using K nearest neighbors algorithm. This method works better than
        # replacing with the median or average or remving completely the data
        imputer = KNNImputer(n_neighbors=n_neighborhs, weights="uniform")
        X_df = pd.DataFrame(imputer.fit_transform(X_df), columns=np.array(X_df.columns))

        return X_df

    def _feature_names(self, X_df):
        return np.array(X_df.columns)


if __name__ == '__main__':
    etl = ETL(path='./Data/titanic/train.csv', list_cat=['Sex', 'Embarked'],
              list_num=['Age', 'SibSp', 'Pclass', 'Parch', 'Fare'], target_label='Survived')

    etl.pipeline()
