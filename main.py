from ETL import ETL
from Models import Models
import streamlit as st
import numpy as np

etl = ETL(path='./Data/titanic/train.csv', list_cat=['Sex', 'Embarked'],
          list_num=['Age', 'SibSp', 'Pclass', 'Parch', 'Fare'], target_label='Survived')

etl.import_data_df()
list_of_features = etl.columns
st.write("Training Data", etl.data_df)
target_label = st.sidebar.multiselect(
    'Choose target label',
    etl.columns,
    'Survived')

list_num = st.sidebar.multiselect(
    'Choose numerical features',
    etl.columns,
    ['Age', 'SibSp', 'Pclass', 'Parch', 'Fare'])

list_cat = st.sidebar.multiselect(
    'Choose categorical features',
    etl.columns,
    ['Sex', 'Embarked'])

test_size = st.sidebar.slider('test size', min_value=0.1, max_value=1.0, value=0.20, step=0.05)

st.write('You selected to train a model with numerical features:', np.array(list_num), np.array(list_cat))
st.write('Your selected target label is:', target_label)
st.write('Your selected test size is ', test_size)

etl.list_cat = list_cat
etl.list_num = list_num
etl.target_label = target_label

etl.clean_data_pipeline()
etl.create_train_test_datasets(test_size=test_size)

st.write(etl.X_test.shape)

if st.sidebar.button("Train models"):
    models = Models()
    models.model_bag(etl.X_train, etl.y_train, etl.X_test, etl.y_test)
    st.bar_chart(data=models.model_bag_df, x='Model name', y='accuracy')
