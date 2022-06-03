import streamlit as st
import pandas as pd
import numpy as np

from apps.resampling import Resampling


def app():
    st.title("Classification")

    train = pd.read_excel('C://Users//HP Omen 15//PycharmProjects//AutoML//dataset//train.xlsx')
    test = pd.read_excel('C://Users//HP Omen 15//PycharmProjects//AutoML//dataset//test.xlsx')
    # out_of_sample = pd.read_excel('out_sample.xlsx')

    if len(train) == 0:
        raise Exception("Please upload dataset first!")
    if len(test) == 0:
        raise Exception("Please upload dataset first!")

    target = st.selectbox("Select target column", train.columns)
    goods_label = st.text_input("Type label for positive values", value=0)
    bads_label = st.text_input("Type label for negative values", value=1)

    st.subheader('Train')
    st.bar_chart(data=train[target].value_counts(), width=0, height=0)
    st.subheader('Test')
    st.bar_chart(data=test[target].value_counts(), width=0, height=0)

    train_df = train.copy()
    test_df = test.copy()

    a = Resampling(dataset=train_df, target_col=target, bads_label=bads_label,
                   goods_label=goods_label)  # initialize our ensemble learning
    # st.write(a.goods, "\n", a.bads)
    ######################################################
    #     There was ratio_start=0.15, ratio_end=0.20     #
    ######################################################
    a.balanced_resamples_ratio(ratio_start=0.85, ratio_end=1.0)  # balance by ratio

    features = st.multiselect('Select features', options=list(train.columns),
                              # default=['GCVP_AVER_DED_SUM', 'PKB_Cred_Cards', 'PKB_Num_90', 'PKB_Num_Loans_L6M',
                                      # 'PKB_Num_Term_Contr', 'PKB_max_Dpd_L12M', 'KOLSEM', 'OPEN_AGE', 'GB']
                              )
    # list(train.columns))
    # features = ['GCVP_AVER_DED_SUM', 'PKB_Cred_Cards', 'PKB_Num_90', 'PKB_Num_Loans_L6M',
    #             'PKB_Num_Term_Contr', 'PKB_max_Dpd_L12M', 'KOLSEM', 'OPEN_AGE', 'GB']

    a.feature_select_balanced_samples(features=features)  # select with given features
    st.header('Ensemble training')
    algorithm = st.selectbox("Select an algorithm", ["gradb", "tree", "adaboost", "randomf", "knn"])
    a.ensemble_train(algorithm=algorithm)

    # st.header('Tables')
    # tables = a.get_tables()
    # for i in range(0, len(tables)):
    #     st.write(tables[i].head())

    st.header('Metrics')
    st.subheader('Train')
    a.ensemble_predict(train_df)
    a.upload_data(sample_type='train')
    st.subheader('Test')
    a.ensemble_predict(test_df)
    a.upload_data(sample_type='test')

    a.ensemble_metrics()
