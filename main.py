from ETL import ETL
from Models import Models
import streamlit as st
import matplotlib.pyplot as plt

etl = ETL(path='./Data/titanic/train.csv')
etl.import_data_df()

############# STANDARD SIDEBAR WIDGETS #######

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

############## PREPROCESS ##################
etl.preprocess_pipeline(list_cat, list_num, target_label, test_size)

############## TRAIN #######################

if st.sidebar.button("Train models"):
    models = Models()
    models.model_bag(etl.X_train, etl.y_train, etl.X_test, etl.y_test)
    models.feature_importance(etl.feature_names, models.rf)
    st.write('### Model comparison')
    st.bar_chart(data=models.model_bag_df, x='Model name', y='accuracy')
    st.write('### Feature importance')
    fig = plt.figure()
    plt.barh(models.feature_scores_df['Feature name'], width=models.feature_scores_df['Feature score'], align='center')
    st.pyplot(fig)
else:
    st.write("### Raw Data ( .csv)", etl.data_df)