import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

st.set_page_config(page_title="Scope 3 Estimation", page_icon="📈")

st.markdown("# Estimating scope 3 Emmissions")
st.sidebar.header("Scope 3 Estimation")

# Setting Up Linear Regression Model

client_measure_df = pd.read_csv('Client Measurement Data Fix v1.0.csv')

client_measure_df.drop(client_measure_df[client_measure_df['Scope 1 total KGCO2e'] == 0].index, inplace = True)


client_measure_df= client_measure_df.replace('-',np.nan)
client_measure_df= client_measure_df.replace('',np.nan)
client_measure_df= client_measure_df.replace(' ',np.nan)
client_measure_df = client_measure_df.apply(lambda x: pd.to_numeric(x,errors='ignore'))
client_measure_df['# Sites'] = client_measure_df['# Sites'].fillna(1)
client_measure_df['# Employees'] = client_measure_df['# Employees'].fillna(0)
client_measure_df['Measurement No.'] = client_measure_df['Measurement No.'].fillna(1)

pd.set_option('display.float_format', '{:.2f}'.format)

drop_scope_df =  client_measure_df.dropna(subset=['Scope 1 total KGCO2e'])
new_df = client_measure_df.dropna(subset=['Stationary Combustion'])


drop_empty_df = new_df.dropna(subset=['Sector / Industry','Measurement No.','Client Name','Reporting Year End Date','# Employees'])
drop_lease_franchise_processdf = drop_empty_df.drop(columns=['Leased Assets', 'Leased Assets .','Franchises','Processing of Sold Products','Use of Sold Products','Investments'])

clean_df=drop_lease_franchise_processdf.dropna(subset=['Purchased Electricity'])

goodServ = clean_df['Purchased Goods & Services']
scope3tot = clean_df['Scope 3 total KGCO2e']

goodServ_np = goodServ.to_numpy()
scope3tot_np = scope3tot.to_numpy()

sklearn_model = LinearRegression().fit(goodServ_np.reshape((109,1)), scope3tot_np)
sklearn_scope3_predictions = sklearn_model.predict(goodServ_np.reshape((109, 1)))

number = st.number_input('Insert Purchased Goods & Services Value ')
st.write('The current number Purchased Goods & Services is ', number)

first_row = np.array([number])

scope3_user_pred = sklearn_model.predict(first_row.reshape(1, -1))
st.write('The estimated scope 3 Value is ', scope3_user_pred.flat[0] )


cliemtname = clean_df['Client Name']
goodServ_s = clean_df['Purchased Goods & Services']
scope3tot_s =  clean_df['Scope 3 total KGCO2e']


scope3_predictions_df = pd.DataFrame({'Client Name':cliemtname,'Purchased Goods & Services':goodServ_s,
                               'Scope 3 total KGCO2e':scope3tot_s,
                               'Sklearn scope3 Predictions':sklearn_scope3_predictions})

scope3_predictions_df['change'] = (scope3_predictions_df['Sklearn scope3 Predictions'] - scope3_predictions_df['Scope 3 total KGCO2e'])/ scope3_predictions_df['Scope 3 total KGCO2e'] * 100


y_true = scope3_predictions_df['Scope 3 total KGCO2e'].values.tolist()
y_pred = scope3_predictions_df['Sklearn scope3 Predictions'].values.tolist()

st.write("The mean absolute error for data used is: " ,mean_absolute_error(y_true, y_pred))
st.write("The mean absolute error for data used percentage is: " ,mean_absolute_percentage_error(y_true, y_pred))


with st.expander("See Model Data"):
    st.dataframe(scope3_predictions_df)

st.markdown("### Explanation of Scope 3 results")
st.write(
    """A summary of the linear regression model for estimating scope 3 emmissions using Purchased Goods & Services.

The accuracy of predicting scope 3 emmisions based on Purchased Goods & Services is very low. 

Mean absolute Error: 1290756.1170379505

In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 loss function. As can be seen the mean absolute error for this model is extremely high. This model would not be applicable to be used as real world data would not produce accurate results.


Mean Absolute percentage Error:1.2450108947679812

Mean absolute percentage error measures the average magnitude of error produced by a model, or how far off predictions are on average. A MAPE value of 1.2450108947679812 means that the average absolute percentage difference between the predictions and the actuals is 1.2450108947679812. Although this percentage is low , the scores for MSE & and MAE are highly inaccurate. Suggesting predictions produced by this model are not reliable."""
)