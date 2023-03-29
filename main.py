from ETL import ETL
from Models import Models
import streamlit as st
import plotly.graph_objects as go

etl = ETL(path='./Data/titanic/train.csv')
etl.import_data_df()

############# STANDARD SIDEBAR WIDGETS #######

target_label = st.sidebar.multiselect(
    'Choose target label',
    etl.columns)

list_num = st.sidebar.multiselect(
    'Choose numerical features',
    etl.columns)

list_cat = st.sidebar.multiselect(
    'Choose categorical features',
    etl.columns)

test_size = st.sidebar.slider('test size', min_value=0.1, max_value=1.0, value=0.20, step=0.05)

############## PREPROCESS ##################
etl.preprocess_pipeline(list_cat, list_num, target_label, test_size)

############## TRAIN #######################

if st.sidebar.button("Train models"):
    models = Models()
    models.model_bag(etl.X_train, etl.y_train, etl.X_test, etl.y_test)
    models.feature_importance(etl.feature_names, models.rf)
    st.write('### Model comparison')

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=models.model_bag_df['Model name'],
        y=models.model_bag_df['accuracy'],
        name='accuracy',
        orientation='v',
    ))
    st.plotly_chart(fig1)
    st.write('### Feature importance')

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        y=models.feature_scores_df['Feature name'],
        x=models.feature_scores_df['Feature score'],
        name='',
        orientation='h',
    ))
    st.plotly_chart(fig2)

else:
    st.write("### Raw Data ( .csv)", etl.data_df)