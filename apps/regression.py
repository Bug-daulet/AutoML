import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import r2_score

from apps.resampling import Resampling


def app():
    st.title("Regression")
    train = pd.read_excel('C://Users//HP Omen 15//PycharmProjects//AutoML//dataset//train.xlsx')
    test = pd.read_excel('C://Users//HP Omen 15//PycharmProjects//AutoML//dataset//test.xlsx')

    if len(train) == 0:
        raise Exception("Please upload dataset first!")
    if len(test) == 0:
        raise Exception("Please upload dataset first!")

    target = st.selectbox("Select target column", train.columns)

    x_train = train.drop([target], axis=1)
    y_train = train[target]

    x_test = test.drop([target], axis=1)
    y_test = test[target]

    algorithm = st.selectbox("Select an algorithm", ["Linear Regression", "Random forest"])

    if algorithm == "Linear Regression":
        regressor = LinearRegression()
    else:
        regressor = RandomForestRegressor(n_estimators = 100,
                                          criterion = 'mse',
                                          random_state = 1,
                                          n_jobs = -1)

    regressor.fit(x_train, y_train)

    y_predict = regressor.predict(x_test)

    st.header("")
    plt.scatter(y_test, y_predict)
    plt.xlabel("Y values")
    plt.ylabel("Predictions")
    st.pyplot(plt)

    st.header("Metrics")
    st.write('R2 Score for Linear Regression on test data: {}'.format(np.round(r2_score(y_test, y_predict), 3)))
    st.write('MAE:', metrics.mean_absolute_error(y_test, y_predict))
    st.write('MSE:', metrics.mean_squared_error(y_test, y_predict))
    st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
