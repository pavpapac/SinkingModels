from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from ETL import ETL
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression


class Models:

    def __init__(self):
        self.lgr = LogisticRegression()
        self.rf = RandomForestClassifier()
        self.ebm = ExplainableBoostingClassifier()
        self.gbc = GradientBoostingClassifier()
        self.init_models = [self.lgr, self.rf, self.gbc, self.ebm]
        self.model_bag_df = pd.DataFrame(columns=['Model name', 'Trained estimator', 'accuracy'])
        self.model_bag_df['Model name'] = ['Logistic Regr', 'Random Forest', 'Explain Boost', 'Grad Boost']
        self.feature_scores_df = pd.DataFrame(columns=['Feature name', 'Feature score'])

    def model_bag(self, X_train, y_train, X_test, y_test):
        # Create a bag with trained estimators and their reported accuracy

        test_acc = []
        trained_models = []
        for model in self.init_models:
            trained_model = self.train_model(model, X_train, y_train)
            accuracy = self.test_model(model, X_test, y_test)
            trained_models.append(trained_model)
            test_acc.append(accuracy)
        self.model_bag_df['Trained estimator'] = trained_models
        self.model_bag_df['accuracy'] = test_acc

    def train_model(self, estimator, X_train, y_train):
        # Given an estimator, train on the train set

        estimator.fit(X_train, y_train)

        return estimator

    def test_model(self, estimator, X_test, y_test):
        # Given a trained estimator, make prediction and check accuracy on test set

        y_pred = estimator.predict(X_test)
        y_pred = y_pred.reshape(-1, 1)
        acc = np.sum(y_pred == y_test) / np.size(y_pred == y_test)

        return acc

    def feature_importance(self, feature_names, estimator):

        self.feature_scores_df['Feature name'] = feature_names
        self.feature_scores_df['Feature score'] = estimator.feature_importances_
        self.feature_scores_df.sort_values(by='Feature score', ascending=True, inplace=True)

if __name__ == "__main__":
    etl = ETL(path='./Data/titanic/train.csv')
    etl.import_data_df()
    list_cat = ['Sex', 'Embarked']
    list_num = ['Age', 'SibSp', 'Pclass', 'Parch', 'Fare']
    target_label = ['Survived']
    etl.preprocess_pipeline(list_cat, list_num, target_label)
    models = Models()
    models.model_bag(etl.X_train, etl.y_train, etl.X_test, etl.y_test)
    models.feature_importance(etl.feature_names, models.rf)
    print("Accuracy per model: ", models.model_bag_df)
    print("Feature importance ", models.feature_scores_df)