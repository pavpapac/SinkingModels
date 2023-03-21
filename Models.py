from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import numpy as np
import pandas as pd
from ETL import ETL

class Models:

    def __init__(self):
        self.rf = []
        self.acc = []
        self.feat_importance_df = pd.DataFrame(columns=['feature', 'importance'])


    def train_forest(self, X_train, y_train, n_estimators, max_features='sqrt'):
        self.rf = RandomForestClassifier(n_estimators, max_features=max_features)
        self.rf.fit(X_train, y_train)

    def test_forest(self, X_test, y_test):
        y_pred = self.rf.predict(X_test)
        self.acc = np.sum(y_pred == y_test)/len(y_pred == y_test)


    def rank_feature_importance(self, feature_names):
        self.feat_importance_df['feature'] = feature_names
        self.feat_importance_df['importance'] = self.rf.feature_importances_
        self.feat_importance_df.sort_values(by='importance', ascending=True, inplace=True)

if __name__ == "__main__":
    etl = ETL(path='./Data/titanic/train.csv', list_cat=['Sex', 'Embarked'],
                       list_num=['Age', 'SibSp', 'Pclass', 'Parch', 'Fare'], target_label='Survived')
    etl.pipeline()
    models = Models()
    models.train_forest(etl.X_train, etl.y_train, n_estimators=200, max_features='sqrt')
    models.test_forest(etl.X_test, etl.y_test)
    models.rank_feature_importance(etl.feature_names)
    print(models.acc)


